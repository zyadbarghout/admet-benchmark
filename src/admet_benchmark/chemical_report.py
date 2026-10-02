"""Interactive chemical-space reports assembled from completed benchmark runs.

The report generator consumes saved artifacts only. It does not refit models or
recalculate benchmark metrics, so exploratory filtering cannot leak into model
selection or the locked test evaluation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from io import BytesIO
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from plotly.offline import get_plotlyjs
from rdkit import Chem, rdBase
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .features import build_features
from .tables import read_csv
from .chemical_space import build_map_atlas, GTMConfig


@dataclass(frozen=True)
class RunArtifacts:
    endpoint: dict[str, Any]
    records: list[dict[str, Any]]


# These are intentionally recognisable medicinal-chemistry motifs rather than
# mined fragments with unstable atom numbering. The report shows the ten most
# frequent motifs in the displayed chemical space.
_CURATED_SUBSTRUCTURES = (
    ("Aromatic ring", "[a;r5,r6]"),
    ("Phenyl ring", "c1ccccc1"),
    ("Heteroaromatic atom", "[a;!#6]"),
    ("Amide", "[NX3][CX3](=[OX1])"),
    ("Carboxylic acid", "[CX3](=[OX1])[OX2H1]"),
    ("Ester", "[CX3](=[OX1])[OX2][#6]"),
    ("Non-amide amine", "[NX3;!$(N[C,S]=[O,S,N])]"),
    ("Hydroxyl", "[OX2H]"),
    ("Ether", "[OD2]([#6])[#6]"),
    ("Halogen", "[F,Cl,Br,I]"),
    ("Nitrile", "[CX2]#N"),
    ("Sulfonamide", "S(=O)(=O)N"),
    ("Urea", "N[C](=O)N"),
    ("Thioether", "[SX2]([#6])[#6]"),
    ("Phosphate", "P(=O)(O)O"),
)


def generate_chemical_report(
    run_dirs: Iterable[str | Path],
    *,
    output_path: str | Path,
    max_points: int = 6_000,
    perplexity: float = 35.0,
    seed: int = 42,
    embedding: str = "tsne",
    map_methods: Iterable[str] | None = None,
    representations: Iterable[str] | None = None,
    map_cache: str | Path | None = None,
    gtm_config: GTMConfig | None = None,
    smarts_patterns: Iterable[str] | None = None,
    delivery: str = "inline",
) -> Path:
    """Generate one multi-endpoint, molecule-aware HTML report.

    Every evaluation observation is kept whenever ``max_points`` permits it.
    Development records are sampled after evaluation records, because they are
    context for the chemical map rather than scored observations.
    """

    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    if not paths:
        raise ValueError("At least one benchmark run directory is required")
    if embedding not in {"tsne", "pca", "gtm"}:
        raise ValueError("embedding must be tsne, gtm or pca")
    if not np.isfinite(perplexity) or perplexity <= 0:
        raise ValueError("perplexity must be finite and positive")
    if max_points < 3:
        raise ValueError("max_points must be at least three")

    # Some valid salts and edge-case structures make RDKit emit warnings while
    # records, descriptors, or depictions are built. Invalid inputs are still
    # rejected; keep report generation quiet without weakening validation.
    with rdBase.BlockLogs():
        bundles = [bundle for path in paths for bundle in _load_run(path)]
        endpoint_ids = [bundle.endpoint["id"] for bundle in bundles]
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("Endpoint labels must be unique across combined runs")

        all_records = [record for bundle in bundles for record in bundle.records]
        records = _sample_records(all_records, max_points=max_points, seed=seed)
        atlas = None
        if map_methods or representations or embedding == "gtm":
            atlas = build_map_atlas(records, methods=map_methods or (embedding,),
                representations=representations or ("morgan_rdkit",), seed=seed,
                perplexity=perplexity, cache_dir=map_cache, gtm_config=gtm_config)
        if atlas and atlas["views"]:
            first_view = next((view for view in atlas["views"] if view["method"] == embedding), atlas["views"][0])
            coordinates = dict(zip(atlas["chemical_ids"], first_view["coordinates"], strict=True))
            for record in records:
                record["x"], record["y"] = coordinates[record["chemical_id"]]
        else:
            _attach_embedding(records, perplexity=perplexity, seed=seed, method="pca" if embedding == "gtm" else embedding)
        filters = _attach_substructures(records, smarts_patterns or [])
        structures = _attach_structure_images(records)

    dashboard = {
        "title": "Chemical Model Diagnostics",
        "embedding": "Switchable chemical atlas" if atlas and atlas["views"] else f"{'PCA' if embedding in {'pca', 'gtm'} else 't-SNE'} on Morgan fingerprints and scaled RDKit descriptors",
        "embedding_axes": first_view["axes"] if atlas and atlas["views"] else [f"{'PC' if embedding in {'pca', 'gtm'} else 't-SNE'} {i}" for i in (1, 2)],
        "embedding_seed": seed,
        "evidence_level": "Exploratory diagnostics",
        "atlas": atlas,
        "records": records,
        "structures": structures,
        "endpoints": [bundle.endpoint for bundle in bundles],
        "filters": filters,
        "source_runs": [_source_run_provenance(path) for path in paths],
        "total_points": len(all_records),
        "sampled_points": len(records),
        "sampling_applied": len(records) < len(all_records),
        "evaluation_points": sum(record["split"] != "development" for record in records),
    }
    dashboard["report_signature"] = hashlib.sha256(
        json.dumps(dashboard, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if delivery == "assets":
        from .report_delivery import write_assets_report
        write_assets_report(dashboard, destination)
    elif delivery == "inline":
        destination.write_text(_report_html(dashboard), encoding="utf-8")
    else:
        raise ValueError("delivery must be inline or assets")
    return destination


_PROVENANCE_FILES = (
    "predictions.csv",
    "fold_metrics.csv",
    "pairwise_comparisons.csv",
    "validation_report.json",
    "benchmark_metadata.json",
    "resolved_config.yaml",
    "model_comparison.csv",
    "validation_predictions.csv",
    "held_out_test_predictions.csv",
    "evaluation/fold_metrics.csv",
)


def _source_run_provenance(run: Path) -> dict[str, Any]:
    """Return portable source identity without leaking a host filesystem path."""

    digest = hashlib.sha256()
    artifacts: list[str] = []
    for relative_name in _PROVENANCE_FILES:
        artifact = run / relative_name
        if not artifact.is_file():
            continue
        artifacts.append(relative_name)
        digest.update(relative_name.encode("utf-8"))
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "name": run.name,
        "signature": digest.hexdigest()[:16],
        "artifact_count": len(artifacts),
    }


def _load_run(run: Path) -> list[RunArtifacts]:
    """Load either a canonical evaluation bundle or a legacy benchmark run."""

    if all((run / name).exists() for name in ("predictions.csv", "fold_metrics.csv")):
        return _load_evaluation_bundle(run)
    return [_load_benchmark_run(run)]


def _load_benchmark_run(run: Path) -> RunArtifacts:
    required = (
        "benchmark_metadata.json",
        "resolved_config.yaml",
        "model_comparison.csv",
        "data/development.csv",
        "validation_predictions.csv",
        "held_out_test_predictions.csv",
    )
    missing = [name for name in required if not (run / name).exists()]
    if missing:
        raise FileNotFoundError(f"{run} is missing benchmark artifacts: {', '.join(missing)}")

    metadata = json.loads((run / "benchmark_metadata.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    comparison = read_csv(run / "model_comparison.csv")
    target = str(metadata["target"])
    configured_label = config.get("project_name")
    run_name = str(config.get("run_name") or target)
    if configured_label:
        label = str(configured_label).removesuffix(" model arena")
    elif run_name.lower().startswith("openadmet_"):
        label = target
    elif "bbbp" in run_name.lower():
        label = "BBBP"
    else:
        label = run_name.replace("_", " ")
    label = label.strip()
    endpoint_id = hashlib.sha1(_source_run_provenance(run)["signature"].encode("utf-8")).hexdigest()[:12]
    models = comparison["model"].astype(str).tolist()
    smiles_column = str(config["dataset"]["smiles_column"])
    metric_path = run / "evaluation" / "fold_metrics.csv"
    metrics = read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    preprocessing = (
        str(metrics["preprocessing"].dropna().iloc[0])
        if "preprocessing" in metrics and metrics["preprocessing"].notna().any()
        else "benchmark"
    )
    split_strategy = (
        str(metrics["split_strategy"].dropna().iloc[0])
        if "split_strategy" in metrics and metrics["split_strategy"].notna().any()
        else "provided"
    )
    if "preprocessing" not in metrics:
        metrics["preprocessing"] = preprocessing
    if "split_strategy" not in metrics:
        metrics["split_strategy"] = split_strategy

    records: list[dict[str, Any]] = []
    frames = (
        ("development", read_csv(run / "data/development.csv")),
        ("validation", read_csv(run / "validation_predictions.csv")),
        ("test", read_csv(run / "held_out_test_predictions.csv")),
    )
    for split, frame in frames:
        records.extend(
            _frame_records(
                frame,
                endpoint_id=endpoint_id,
                endpoint_label=label,
                task=str(metadata["task"]),
                split=split,
                smiles_column=smiles_column,
                models=models,
                preprocessing=preprocessing,
                split_strategy=split_strategy,
            )
        )

    threshold = float(
        dict(config.get("benchmark", {}).get("evaluation") or {}).get(
            "decision_threshold", 0.5
        )
    )
    endpoint = {
        "id": endpoint_id,
        "label": label,
        "target": target,
        "task": str(metadata["task"]),
        "models": models,
        "selected_model": str(metadata["selected_model"]),
        "selection_metric": str(metadata["selection_metric"]),
        "threshold": threshold,
        "target_transform": str(config["dataset"].get("target_transform", "none")),
        "comparison": _json_records(comparison),
        "metrics": _json_records(metrics),
        "split": metadata.get("split", {}),
        "validation_split": metadata.get("validation_split", {}),
        "run_dir": run.name,
        "interval_confidence": config.get("uncertainty", {}).get("interval_confidence"),
            "cv_evidence": config.get("cv_evidence", False),
            "uq_description": config.get("uq_description", "Submitted uncertainty; see source protocol"),
    }
    return RunArtifacts(endpoint=endpoint, records=records)


def _load_evaluation_bundle(run: Path) -> list[RunArtifacts]:
    """Adapt the evaluator's stable long-form records to the chemical report."""

    predictions = read_csv(run / "predictions.csv")
    metrics = read_csv(run / "fold_metrics.csv")
    config_path = run / "resolved_config.yaml"
    config = (
        yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config_path.exists()
        else {}
    )
    required = {
        "task",
        "endpoint",
        "model",
        "partition",
        "repeat",
        "fold",
        "sample_id",
        "molecule_id",
        "smiles",
        "observed",
        "prediction",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(
            f"{run} canonical predictions are missing columns: {', '.join(missing)}"
        )

    bundles: list[RunArtifacts] = []
    canonical_smiles_cache: dict[str, str | None] = {}
    endpoint_values = predictions[["task", "endpoint"]].drop_duplicates()
    for task, target in endpoint_values.itertuples(index=False, name=None):
        subset = predictions[
            predictions["task"].astype(str).eq(str(task))
            & predictions["endpoint"].astype(str).eq(str(target))
        ].copy()
        endpoint_id = hashlib.sha1(
            f"{_source_run_provenance(run)['signature']}:{task}:{target}".encode("utf-8")
        ).hexdigest()[:12]
        models = sorted(subset["model"].astype(str).unique())
        label = str(config.get("display_name") or target)
        records: list[dict[str, Any]] = []
        group_columns = [
            "preprocessing",
            "split_strategy",
            "partition",
            "repeat",
            "fold",
            "sample_id",
        ]
        for keys, group in subset.groupby(group_columns, dropna=False, sort=True):
            preprocessing, split_strategy, partition, repeat, fold, sample_id = keys
            first = group.iloc[0]
            submitted_smiles = str(first["smiles"])
            if submitted_smiles not in canonical_smiles_cache:
                molecule = Chem.MolFromSmiles(submitted_smiles)
                canonical_smiles_cache[submitted_smiles] = (
                    Chem.MolToSmiles(molecule, isomericSmiles=True)
                    if molecule is not None else None
                )
            canonical = canonical_smiles_cache[submitted_smiles]
            if canonical is None:
                continue
            model_predictions: dict[str, dict[str, float]] = {}
            for _, row in group.iterrows():
                model = str(row["model"])
                prediction = float(row["prediction"])
                observed = float(row["observed"])
                result = {
                    "prediction": prediction,
                    "residual": prediction - observed,
                    "absolute_error": abs(prediction - observed),
                }
                for name in (
                    "uncertainty",
                    "uncertainty_percentile",
                    "prediction_lower",
                    "prediction_upper",
                    "threshold",
                    "absolute_lower", "absolute_upper", "ensemble_std", "set_0", "set_1",
                    "nearest_train_similarity", "novel_scaffold", "property_cliff",
                    "max_similar_property_gap", "nominal_coverage", "raw_probability",
                ):
                    if name in row and pd.notna(row[name]):
                        result[name] = float(row[name])
                if "nearest_train_id" in row and pd.notna(row["nearest_train_id"]):
                    result["nearest_train_id"] = str(row["nearest_train_id"])
                model_predictions[model] = result
            partition_name = str(partition)
            display_split = "development" if partition_name == "train" else partition_name
            record_id = (
                f"{endpoint_id}:{preprocessing}:{split_strategy}:"
                f"{display_split}:{repeat}:{fold}:{sample_id}"
            )
            records.append(
                {
                    "id": record_id,
                    "chemical_id": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16],
                    "endpoint": endpoint_id,
                    "endpoint_label": label,
                    "task": str(task),
                    "split": display_split,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "preprocessing": str(preprocessing),
                    "split_strategy": str(split_strategy),
                    "sample_id": str(sample_id),
                    "molecule_id": str(first["molecule_id"]),
                    "original_sample_id": str(first.get("original_sample_id", "")),
                    "smiles": str(first["smiles"]),
                    "canonical_smiles": canonical,
                    "target_raw": float(
                        first.get("observed_submitted", first["observed"])
                    ),
                    "target_model": float(first["observed"]),
                    "predictions": model_predictions,
                    "structure": None,
                    "substructures": [],
                }
            )

        endpoint_metrics = metrics[
            metrics["task"].astype(str).eq(str(task))
            & metrics["endpoint"].astype(str).eq(str(target))
        ].copy()
        threshold_values = subset.get("threshold", pd.Series(dtype=float)).dropna()
        transform_values = subset.get(
            "target_transform", pd.Series(["none"])
        ).dropna()
        split_values = subset.get(
            "split_strategy", pd.Series(["provided"])
        ).dropna().astype(str).unique()
        endpoint = {
            "id": endpoint_id,
            "label": label,
            "evaluation_units": config.get("evaluation_units", "evaluation-scale units"),
            "target": str(target),
            "task": str(task),
            "models": models,
            "selected_model": str(config.get("selected_model") or models[0]),
            "selection_metric": "not selected by evaluator",
            "threshold": float(threshold_values.iloc[0]) if len(threshold_values) else 0.5,
            "target_transform": str(transform_values.iloc[0]) if len(transform_values) else "none",
            "comparison": [],
            "metrics": _json_records(endpoint_metrics),
            "split": {"strategy": ", ".join(sorted(split_values))},
            "validation_split": {"strategy": "provided fold assignments"},
            "run_dir": run.name,
            "interval_confidence": config.get("uncertainty", {}).get("interval_confidence"),
            "cv_evidence": config.get("cv_evidence", False),
            "uq_description": config.get("uq_description", "Submitted uncertainty; see source protocol"),
        }
        bundles.append(RunArtifacts(endpoint=endpoint, records=records))
    return bundles


def _frame_records(
    frame: pd.DataFrame,
    *,
    endpoint_id: str,
    endpoint_label: str,
    task: str,
    split: str,
    smiles_column: str,
    models: list[str],
    preprocessing: str,
    split_strategy: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_number, row in frame.iterrows():
        molecule = Chem.MolFromSmiles(str(row[smiles_column]))
        if molecule is None:
            continue
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
        sample_id = str(row.get("sample_id", f"row-{row_number}"))
        predictions: dict[str, dict[str, float]] = {}
        if split != "development":
            observed = float(row["target_model"])
            for model in models:
                value = row.get(f"prediction__{model}")
                if value is None or pd.isna(value):
                    continue
                prediction = float(value)
                residual = prediction - observed
                predictions[model] = {
                    "prediction": prediction,
                    "residual": residual,
                    "absolute_error": abs(residual),
                }
        record_id = f"{endpoint_id}:{split}:{sample_id}:{row_number}"
        records.append(
            {
                "id": record_id,
                "chemical_id": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16],
                "endpoint": endpoint_id,
                "endpoint_label": endpoint_label,
                "task": task,
                "split": split,
                "repeat": 0,
                "fold": 0,
                "preprocessing": preprocessing,
                "split_strategy": split_strategy,
                "sample_id": sample_id,
                "molecule_id": str(row.get("molecule_id", sample_id)),
                "smiles": str(row[smiles_column]),
                "canonical_smiles": canonical,
                "target_raw": float(row["target_raw"]),
                "target_model": float(row["target_model"]),
                "predictions": predictions,
                "structure": None,
                "substructures": [],
            }
        )
    return records


def _sample_records(
    records: list[dict[str, Any]], *, max_points: int, seed: int
) -> list[dict[str, Any]]:
    """Downsample whole chemical entities so fold/endpoint histories stay intact."""

    if len(records) <= max_points:
        return records
    evaluation = [record for record in records if record["split"] != "development"]
    development = [record for record in records if record["split"] == "development"]
    rng = np.random.default_rng(seed)

    def choose_complete_chemicals(
        candidates: list[dict[str, Any]], capacity: int
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in candidates:
            grouped.setdefault(str(record["chemical_id"]), []).append(record)
        identifiers = np.array(sorted(grouped), dtype=object)
        rng.shuffle(identifiers)
        selected: list[dict[str, Any]] = []
        for identifier in identifiers:
            group = grouped[str(identifier)]
            if len(selected) + len(group) > capacity:
                continue
            selected.extend(group)
            if len(selected) == capacity:
                break
        if not selected and grouped:
            # ``max_points`` is a soft ceiling when a single molecule owns more
            # observations than the limit. Preserving its complete history is
            # safer than silently slicing its folds or endpoints.
            return min(grouped.values(), key=len)
        return selected

    sampled_evaluation = choose_complete_chemicals(evaluation, max_points)
    if len(sampled_evaluation) >= max_points or len(evaluation) > max_points:
        return sampled_evaluation
    remaining = max_points - len(sampled_evaluation)
    return sampled_evaluation + choose_complete_chemicals(development, remaining)


def _attach_embedding(
    records: list[dict[str, Any]], *, perplexity: float, seed: int, method: str = "tsne"
) -> None:
    canonical = list(dict.fromkeys(record["canonical_smiles"] for record in records))
    if not canonical:
        raise ValueError("No valid molecular observations to display")
    if len(canonical) <= 2:
        positions = {smiles: (float(i), 0.0) for i, smiles in enumerate(canonical)}
        for record in records:
            record["x"], record["y"] = positions[record["canonical_smiles"]]
        return
    features = build_features(
        canonical,
        representation="morgan_rdkit",
        descriptor_set="compact",
        n_bits=1024,
    ).values
    features = SimpleImputer(strategy="median").fit_transform(features)
    features = StandardScaler().fit_transform(features)
    components = min(50, features.shape[1], len(canonical) - 1)
    reduced = PCA(
        n_components=components, svd_solver="randomized", random_state=seed
    ).fit_transform(features)
    valid_perplexity = min(float(perplexity), max(2.0, (len(canonical) - 1) / 3))
    if method == "pca":
        coordinates = reduced[:, :2]
    else:
        coordinates = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=valid_perplexity,
            random_state=seed,
            n_jobs=-1,
        ).fit_transform(reduced)
    positions = {
        smiles: (float(point[0]), float(point[1]))
        for smiles, point in zip(canonical, coordinates, strict=True)
    }
    for record in records:
        record["x"], record["y"] = positions[record["canonical_smiles"]]


def _attach_substructures(
    records: list[dict[str, Any]], custom_patterns: Iterable[str]
) -> list[dict[str, Any]]:
    candidates = list(_CURATED_SUBSTRUCTURES)
    for pattern in custom_patterns:
        if Chem.MolFromSmarts(pattern) is None:
            raise ValueError(f"Invalid SMARTS pattern: {pattern!r}")
        candidates.append((pattern, pattern))

    unique_molecules = {
        smiles: Chem.MolFromSmiles(smiles)
        for smiles in dict.fromkeys(record["canonical_smiles"] for record in records)
    }
    matches: dict[str, set[str]] = {smiles: set() for smiles in unique_molecules}
    filters: list[dict[str, Any]] = []
    for label, smarts in candidates:
        query = Chem.MolFromSmarts(smarts)
        assert query is not None
        key = hashlib.sha1(smarts.encode("utf-8")).hexdigest()[:12]
        count = 0
        for smiles, molecule in unique_molecules.items():
            if molecule is not None and molecule.HasSubstructMatch(query):
                matches[smiles].add(key)
                count += 1
        if count:
            filters.append({"key": key, "label": label, "smarts": smarts, "count": count})
    filters.sort(key=lambda item: (-int(item["count"]), str(item["label"])))
    filters = filters[:10]
    retained = {item["key"] for item in filters}
    for record in records:
        record["substructures"] = sorted(matches[record["canonical_smiles"]] & retained)
    return filters


def _attach_structure_images(records: list[dict[str, Any]]) -> dict[str, str]:
    """Return compact, deduplicated structure depictions keyed by chemistry."""

    images: dict[str, str] = {}
    for record in records:
        key = record["chemical_id"]
        if key not in images:
            molecule = Chem.MolFromSmiles(record["canonical_smiles"])
            if molecule is None:
                images[key] = ""
            else:
                image = Draw.MolToImage(molecule, size=(260, 180))
                buffer = BytesIO()
                image.save(buffer, format="WEBP", quality=72, method=4)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                images[key] = f"data:image/webp;base64,{encoded}"
        record["structure"] = key
    return images


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cleaned = frame.replace({np.nan: None})
    return cleaned.to_dict(orient="records")


def _report_html(dashboard: dict[str, Any]) -> str:
    data = json.dumps(dashboard, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return (
        (Path(__file__).parent / "report_assets" / "report.html").read_text(encoding="utf-8")
        .replace("__TITLE__", escape(str(dashboard["title"])))
        .replace("__CSS__", (Path(__file__).parent / "report_assets" / "report.css").read_text(encoding="utf-8"))
        .replace("__JS__", (Path(__file__).parent / "report_assets" / "report.js").read_text(encoding="utf-8"))
        .replace("__SCIENTIFIC_JS__", (Path(__file__).parent / "report_assets" / "scientific.js").read_text(encoding="utf-8"))
        .replace("__WORKSPACE_JS__", (Path(__file__).parent / "report_assets" / "workspace.js").read_text(encoding="utf-8"))
        .replace("__PLOTLY__", get_plotlyjs())
        .replace("__DATA__", data)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a multi-endpoint interactive chemical-space report."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Completed benchmark directories")
    parser.add_argument("--output", required=True, type=Path, help="HTML report destination")
    parser.add_argument("--delivery", choices=("inline", "assets"), default="inline")
    parser.add_argument("--max-points", type=int, default=6_000)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding", choices=("tsne", "gtm", "pca"), default="tsne")
    parser.add_argument("--map-methods", nargs="+", choices=("tsne", "gtm", "pca"), help="Precompute switchable map techniques")
    parser.add_argument("--representations", nargs="+", choices=("morgan", "rdkit_descriptors", "morgan_rdkit"))
    parser.add_argument("--map-cache", type=Path)
    parser.add_argument("--gtm-nodes", type=int, default=15)
    parser.add_argument("--gtm-rbfs", type=int, default=5)
    parser.add_argument("--gtm-width", type=float, default=1.0)
    parser.add_argument("--gtm-regularization", type=float, default=.01)
    parser.add_argument("--gtm-iterations", type=int, default=150)
    parser.add_argument("--smarts", action="append", default=[])
    args = parser.parse_args()
    result = generate_chemical_report(
        args.run_dirs,
        output_path=args.output,
        max_points=args.max_points,
        perplexity=args.perplexity,
        seed=args.seed,
        embedding=args.embedding,
        map_methods=args.map_methods, representations=args.representations, map_cache=args.map_cache,
        gtm_config=GTMConfig(nodes=args.gtm_nodes, rbfs=args.gtm_rbfs, width=args.gtm_width,
            regularization=args.gtm_regularization, max_iterations=args.gtm_iterations),
        smarts_patterns=args.smarts, delivery=args.delivery,
    )
    print(result)


if __name__ == "__main__":
    main()
