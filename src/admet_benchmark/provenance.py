"""Compatibility contracts for resumable studies and trusted local checkpoints."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from .features import _descriptor_functions


def runtime_signature():
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "rdkit", "joblib")
    return {"python": ".".join(platform.python_version_tuple()[:2]),
            **{name: version(name) for name in packages}}


def implementation_signature():
    root = Path(__file__).parent
    # Include training, evaluation and inference, without invalidating a model
    # when only report styling changes. Filenames bind order and membership.
    names = ("cross_validation.py", "uq.py", "applicability.py", "features.py",
             "models.py", "evaluation.py", "metrics.py", "tables.py",
             "provenance.py", "deployment.py")
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in names if (root / name).exists()}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def feature_schema(representation):
    names = []
    if representation in {"morgan", "morgan_rdkit"}:
        names.extend(f"morgan_{i}" for i in range(1024))
    if representation in {"rdkit_descriptors", "morgan_rdkit"}:
        names.extend(name for name, _ in _descriptor_functions("full"))
    return {"representation": representation, "radius": 2, "n_bits": 1024,
            "descriptor_set": "full", "names": names}


def verify_compatibility(metadata, *, representation=None):
    if metadata.get("runtime") != runtime_signature():
        raise ValueError("Runtime compatibility mismatch; use the recorded environment or retrain in a new registration")
    if metadata.get("implementation_hash") != implementation_signature():
        raise ValueError("Runner implementation changed; create a new registration")
    if representation is not None and metadata.get("feature_schema") != feature_schema(representation):
        raise ValueError("Feature schema mismatch; descriptor names/order or fingerprint settings changed")


def environment_lock():
    path = Path(__file__).resolve().parents[2] / "uv.lock"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
