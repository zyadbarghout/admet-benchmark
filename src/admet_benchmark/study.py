"""Register development protocols and shared, deterministic chemical partitions.

This producer saves repeated holdouts (one fold per repeat), not nested CV.
Fractions refer to chemical groups: the test fraction of all groups, then
validation and calibration fractions of the remaining development groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
import yaml


@dataclass
class PreparedStudy:
    config: dict[str, Any]
    observations: pd.DataFrame
    assignments: pd.DataFrame
    audit: dict[str, Any]


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def prepare_study(path: str | Path) -> PreparedStudy:
    """Read and validate the entire protocol, dataset and partitions without writes."""
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    schema = json.loads((Path(__file__).parent / "schemas/study.schema.json").read_text())
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(config), key=str)
    if errors:
        raise ValueError("Study schema: " + "; ".join(error.message for error in errors))
    endpoints = {e["name"]: e for e in config["endpoints"]}
    if len(endpoints) != len(config["endpoints"]):
        raise ValueError("Endpoint names must be unique")
    if "none" in config["uq_methods"] and len(config["uq_methods"]) > 1:
        raise ValueError("uq_methods cannot combine none with another method")
    for endpoint in endpoints.values():
        classification = endpoint["task"] == "classification"
        valid_metrics = {"roc_auc", "pr_auc", "brier", "log_loss"} if classification else {"mae", "rmse", "r2", "spearman"}
        if endpoint["primary_metric"] not in valid_metrics:
            raise ValueError("Primary metric must match endpoint task")
        if classification and endpoint["transform"] != "none":
            raise ValueError("Classification endpoints require transform: none")
    split = config["split"]
    if split["validation_fraction"] + split["calibration_fraction"] >= 1:
        raise ValueError("Validation and calibration fractions must leave training groups")
    source = Path(config["dataset"]["path"]).expanduser()
    if not source.is_absolute():
        source = path.parent / source
    raw = source.read_bytes()
    if _hash(raw) != config["dataset"]["sha256"]:
        raise ValueError("Dataset SHA-256 does not match the declared study input")
    frame = pd.read_csv(BytesIO(raw), dtype={"sample_id": str, "endpoint": str, "smiles": str})
    required = ["sample_id", "endpoint", "smiles", "observed"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Dataset requires sample_id, endpoint, smiles, observed columns")
    frame = frame[required].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Study observations must be nonempty and complete; omit unmeasured endpoints")
    for column in ("sample_id", "endpoint", "smiles"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Empty {column} is not allowed")
    if set(frame.endpoint) != set(endpoints):
        raise ValueError("Dataset endpoint coverage must exactly match the protocol")
    if frame.duplicated(["endpoint", "sample_id"]).any():
        raise ValueError("Duplicate endpoint/sample identity")
    frame["observed"] = pd.to_numeric(frame.observed, errors="raise")
    if not np.isfinite(frame.observed).all():
        raise ValueError("Observed values must be finite")
    canonical, groups = [], []
    with rdBase.BlockLogs():
        for smiles in frame.smiles:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError("Invalid SMILES in study observations")
            identity = Chem.MolToSmiles(molecule, isomericSmiles=True)
            canonical.append(identity)
            groups.append(identity if split["strategy"] == "molecule" else
                          MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) or "[acyclic]")
    frame["canonical_smiles"] = canonical
    frame["chemical_id"] = [_hash(value.encode()) for value in canonical]
    frame["split_group"] = groups
    if frame.groupby("sample_id").chemical_id.nunique().gt(1).any():
        raise ValueError("A sample_id cannot identify different chemicals across endpoints")
    if frame.duplicated(["endpoint", "chemical_id"]).any():
        raise ValueError("Duplicate chemical entity within endpoint; declare an aggregation policy first")
    for name, endpoint in endpoints.items():
        values = frame.loc[frame.endpoint.eq(name), "observed"]
        if endpoint["task"] == "classification" and set(values) != {0, 1}:
            raise ValueError(f"{name} must contain both binary classes 0 and 1")
        if endpoint["transform"] in {"log10", "ln"} and values.le(0).any():
            raise ValueError(f"{name} requires positive observations for its transform")
    frame = frame.sort_values(["endpoint", "chemical_id", "sample_id"]).reset_index(drop=True)
    unique_groups = sorted(frame.split_group.unique())
    # Hash ordering is deterministic under row permutations and does not inspect labels.
    def ordered(values: list[str], seed: int, stage: str) -> list[str]:
        return sorted(values, key=lambda value: _hash(f"{stage}:{seed}:{value}".encode()))
    test_count = math.ceil(len(unique_groups) * split["test_fraction"])
    test_groups = set(ordered(unique_groups, split["test_seed"], "test")[:test_count])
    development = [g for g in unique_groups if g not in test_groups]
    validation_count = math.ceil(len(development) * split["validation_fraction"])
    calibration_count = math.ceil(len(development) * split["calibration_fraction"])
    if validation_count + calibration_count >= len(development):
        raise ValueError("Too few chemical groups to reserve train, calibration, validation and test")
    assignments, support, signatures = [], [], []
    for repeat, seed in enumerate(split["repeat_seeds"]):
        ordered_groups = ordered(development, seed, "development")
        validation = set(ordered_groups[:validation_count])
        calibration = set(ordered_groups[validation_count:validation_count + calibration_count])
        current = frame[["endpoint", "sample_id", "chemical_id", "split_group"]].copy()
        current["repeat"], current["fold"], current["seed"] = repeat, 0, seed
        current["partition"] = ["test" if g in test_groups else "validation" if g in validation
                                else "calibration" if g in calibration else "train"
                                for g in current.split_group]
        signatures.append(_hash(current[["chemical_id", "partition"]].drop_duplicates().to_csv(index=False).encode()))
        for endpoint, spec in endpoints.items():
            for partition in ("train", "calibration", "validation", "test"):
                selected = current.loc[current.endpoint.eq(endpoint) & current.partition.eq(partition)]
                values = frame.loc[selected.index, "observed"]
                count = len(selected)
                if count < split["minimum_molecules"]:
                    raise ValueError(f"{endpoint}, repeat {repeat}, {partition}: {count} molecules is below minimum support")
                counts = {str(int(k)): int(v) for k, v in values.value_counts().items()} if spec["task"] == "classification" else {}
                if spec["task"] == "classification" and any(counts.get(str(k), 0) < split["minimum_per_class"] for k in (0, 1)):
                    raise ValueError(f"{endpoint}, repeat {repeat}, {partition}: insufficient class support; no automatic seed search performed")
                support.append({"endpoint": endpoint, "repeat": repeat, "partition": partition,
                                "n_molecules": count, "n_groups": selected.split_group.nunique(), "class_counts": counts})
        assignments.append(current)
    warnings = ["Repeated holdouts share data; repeat/fold results are not independent datasets.",
                "Registration does not train models, enforce external test access, or certify confirmatory evidence."]
    if len(set(signatures)) != len(signatures):
        warnings.append("Some repeat seeds produced identical partitions; they are not distinct split repetitions.")
    if split["strategy"] == "scaffold":
        warnings.append("All acyclic molecules form one group. Scaffold separation does not guarantee distance or temporal shift.")
    if len(signatures) < 3:
        warnings.append("Fewer than three repeat seeds: use as a smoke test, not a stability benchmark.")
    audit = {"status": "valid", "study_id": config["study_id"], "evidence_level": config["evidence_level"],
             "dataset_sha256": _hash(raw), "observations": len(frame), "molecules": frame.chemical_id.nunique(),
             "groups": len(unique_groups), "repeats": len(signatures), "distinct_partitions": len(set(signatures)),
             "support": support, "warnings": warnings}
    return PreparedStudy(config, frame, pd.concat(assignments, ignore_index=True), audit)


def freeze_study(path: str | Path, output: str | Path) -> Path:
    """Save a content-hashed registration; refuse to overwrite any existing directory."""
    study = prepare_study(path)
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        # Portable relative dataset reference; retain original hash and source citation.
        config = dict(study.config)
        config["dataset"] = {**config["dataset"], "path": "observations.csv"}
        study.observations[["sample_id", "endpoint", "smiles", "observed"]].to_csv(destination / "observations.csv", index=False)
        config["dataset"]["sha256"] = _hash((destination / "observations.csv").read_bytes())
        (destination / "study.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        study.assignments.to_csv(destination / "fold_assignments.csv", index=False)
        study.observations.to_csv(destination / "molecules.csv", index=False)
        (destination / "preflight.json").write_text(_json(study.audit))
        assignment_hash = _hash((destination / "fold_assignments.csv").read_bytes())
        runs = [{"endpoint": endpoint["name"], "model": model, "repeat": repeat, "fold": 0,
                 "seed": seed, "status": "planned", "assignment_sha256": assignment_hash}
                for endpoint in config["endpoints"] for model in config["models"]
                for repeat, seed in enumerate(config["split"]["repeat_seeds"])]
        (destination / "run_index.json").write_text(_json(runs))
        files = {p.name: _hash(p.read_bytes()) for p in sorted(destination.iterdir())}
        manifest = {"schema_version": 1, "study_id": config["study_id"],
                    "evidence_level": config["evidence_level"], "files": files,
                    "registration_sha256": _hash(_json(files).encode()),
                    "software": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion,
                                 "numpy": np.__version__, "pandas": pd.__version__}}
        (destination / "manifest.json").write_text(_json(manifest))
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def verify_study(path: str | Path) -> dict[str, Any]:
    """Detect edits to registered files and reproduce assignments with this runtime."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    files = manifest["files"]
    required = {"study.yaml", "observations.csv", "fold_assignments.csv", "molecules.csv", "preflight.json", "run_index.json"}
    if set(files) != required or manifest["registration_sha256"] != _hash(_json(files).encode()):
        raise ValueError("Invalid registration manifest")
    for name, expected in files.items():
        if _hash((path / name).read_bytes()) != expected:
            raise ValueError(f"Registered artifact changed: {name}")
    rebuilt = prepare_study(path / "study.yaml")
    if _hash(rebuilt.assignments.to_csv(index=False).encode()) != files["fold_assignments.csv"]:
        raise ValueError("Fold assignments do not reproduce in the current runtime")
    return {"status": "verified", "study_id": manifest["study_id"], "registration_sha256": manifest["registration_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Validate protocol, data and all split supports").add_argument("study", type=Path)
    freeze = commands.add_parser("freeze", help="Save a new registration without overwriting")
    freeze.add_argument("study", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    commands.add_parser("verify", help="Verify hashes and reproduce registered assignments").add_argument("registration", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            result = prepare_study(args.study).audit
        elif args.command == "freeze":
            result = {"registration": str(freeze_study(args.study, args.output))}
        else:
            result = verify_study(args.registration)
        print(_json(result))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"admet-study: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
