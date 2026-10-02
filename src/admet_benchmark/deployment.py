"""Final development-pool fitting with explicitly reserved calibration roles."""
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .features import build_features
from .provenance import feature_schema
from .tables import read_csv
from .uq import ProbabilityCalibrator, regression_calibration, classification_calibration


def refit(study, output):
    from .cross_validation import (
        verify_registration, digest, write_json, inner_holdout, fit_members, _predict,
    )
    study, output = Path(study).resolve(), Path(output).resolve()
    config = verify_registration(study)
    if output.exists():
        raise FileExistsError("Use a fresh deployment directory")
    output.mkdir(parents=True)
    protocol = {
        "registration_sha256": digest(study / "registration.json"),
        "seed": config["seeds"][0], "calibration_strategy": "molecule",
        "calibration_fraction": .2,
        "probability_calibration_fraction_of_remaining": .2,
        "policy": "All curated development chemicals assigned to fitting or disjoint calibration; no historical test labels used",
        "selection": "All registered classical families/features refitted; no winner selected",
        "runtime": config["runtime"], "implementation_hash": config["implementation_hash"],
    }
    write_json(output / "protocol.json", protocol)
    with threadpool_limits(limits=config.get("workers", 2)):
        for spec in config["endpoints"]:
            frame = read_csv(study / spec["name"] / "observations.csv")
            frame.attrs["task"] = spec["task"]
            indices = np.arange(len(frame))
            fit, calibration = inner_holdout(indices, frame, "molecule", .2, config["seeds"][0] + 3000)
            roles = {"fit": fit, "conformal_calibration": calibration}
            if spec["task"] == "classification":
                fit, probability = inner_holdout(fit, frame, "molecule", .2, config["seeds"][0] + 4000)
                roles.update(fit=fit, probability_calibration=probability)
            assignment = frame[["sample_id", "smiles"]].copy()
            assignment["role"] = ""
            for role, index in roles.items():
                assignment.loc[index, "role"] = role
            endpoint_dir = output / spec["name"]
            endpoint_dir.mkdir()
            assignment.to_csv(endpoint_dir / "assignments.csv", index=False)
            y = frame.observed.to_numpy()
            for representation in config["representations"]:
                features = build_features(frame.smiles.tolist(), representation=representation, descriptor_set="full")
                for name in config["models"]:
                    context = {"endpoint": spec["name"], "task": spec["task"], "preprocessing": representation, "model": name}
                    directory = endpoint_dir / representation / name
                    directory.mkdir(parents=True)
                    members = fit_members(features.values[fit], y[fit], spec["task"], name, config, config["seeds"][0])
                    cal_predictions = _predict(members, features.values[calibration], spec["task"])
                    calibrator = None
                    if spec["task"] == "regression":
                        spread = cal_predictions.std(axis=0, ddof=1) if len(members) > 1 else np.zeros(len(calibration))
                        calibrated = regression_calibration(y[calibration], cal_predictions.mean(axis=0), spread,
                            config["confidence"], max(float(np.std(y[fit])) * .05, 1e-6))
                    else:
                        probabilities = _predict(members, features.values[probability], spec["task"]).mean(axis=0)
                        calibrator = ProbabilityCalibrator().fit(probabilities, y[probability])
                        calibrated = classification_calibration(y[calibration], calibrator.predict(cal_predictions.mean(axis=0)), config["confidence"])
                    bundle = {"members": members, "task": spec["task"], "representation": representation,
                        "calibrator": calibrator, "calibration": calibrated, "context": context,
                        "evaluation_scale": spec["evaluation_scale"], "training_smiles": frame.iloc[fit].smiles.tolist(),
                        "training_ids": frame.iloc[fit].sample_id.tolist(), "runtime": config["runtime"],
                        "implementation_hash": config["implementation_hash"], "feature_schema": feature_schema(representation),
                        "model_role": "final development predictor with disjoint reserved calibration; prospective performance unverified"}
                    joblib.dump(bundle, directory / "model.joblib", compress=3)
                    write_json(directory / "calibration.json", calibrated)
                    write_json(directory / "complete.json", {"context": context, "role_counts": {k: len(v) for k, v in roles.items()},
                        "protocol_sha256": digest(output / "protocol.json"),
                        "artifacts": {f: digest(directory / f) for f in ("model.joblib", "calibration.json")}})
                    print(f"Final predictor complete: {spec['name']} / {representation} / {name}", flush=True)
    write_json(output / "manifest.json", {"artifacts": {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()},
        "evidence": "Final models are fitted; OOF performance belongs to the registered study, not to these refits"})
    return output
