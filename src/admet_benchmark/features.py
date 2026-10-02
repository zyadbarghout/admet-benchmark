"""Molecular feature generation used by every model in an experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


@dataclass(frozen=True)
class FeatureSet:
    values: np.ndarray
    names: list[str]


_COMPACT_DESCRIPTORS = {
    "mol_weight": Descriptors.MolWt,
    "log_p": Descriptors.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "h_bond_donors": Lipinski.NumHDonors,
    "h_bond_acceptors": Lipinski.NumHAcceptors,
    "rotatable_bonds": Lipinski.NumRotatableBonds,
    "ring_count": Lipinski.RingCount,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
}


def _descriptor_functions(descriptor_set: str):
    if descriptor_set == "compact":
        return list(_COMPACT_DESCRIPTORS.items())
    functions = list(Descriptors.descList)
    # RDKit's default Ipc is extensive and can exceed float32 by many orders
    # of magnitude. Its documented averaged form preserves the descriptor's
    # information while remaining numerically suitable for ML estimators.
    return [
        (name, (lambda molecule: Descriptors.Ipc(molecule, avg=True)))
        if name == "Ipc"
        else (name, function)
        for name, function in functions
    ]


def _molecule_from_smiles(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES passed to feature generator: {smiles!r}")
    return molecule


def build_features(
    smiles: list[str],
    *,
    radius: int = 2,
    n_bits: int = 1024,
    include_rdkit_descriptors: bool = True,
    representation: str | None = None,
    descriptor_set: str = "compact",
) -> FeatureSet:
    """Create Morgan fingerprints, RDKit 2D descriptors, or both.

    ``include_rdkit_descriptors`` remains as a compatibility option. New
    configurations should use ``representation`` explicitly.
    """

    if radius < 0 or n_bits < 16:
        raise ValueError("radius must be non-negative and n_bits must be at least 16")
    if representation is None:
        representation = "morgan_rdkit" if include_rdkit_descriptors else "morgan"
    representation = representation.lower()
    if representation not in {"morgan", "rdkit_descriptors", "morgan_rdkit"}:
        raise ValueError(
            "features.representation must be morgan, rdkit_descriptors, or morgan_rdkit"
        )
    if descriptor_set not in {"compact", "full"}:
        raise ValueError("features.descriptor_set must be compact or full")

    use_morgan = representation in {"morgan", "morgan_rdkit"}
    use_descriptors = representation in {"rdkit_descriptors", "morgan_rdkit"}
    descriptor_functions = _descriptor_functions(descriptor_set)

    # Some public datasets contain isolated explicit hydrogens. RDKit can read
    # them but emits one warning per molecule, which obscures experiment output.
    RDLogger.DisableLog("rdApp.warning")
    fingerprints = np.zeros((len(smiles), n_bits), dtype=np.float32) if use_morgan else None
    fingerprint_generator = (
        GetMorganGenerator(radius=radius, fpSize=n_bits) if use_morgan else None
    )
    descriptor_rows: list[list[float]] = []
    for row_index, value in enumerate(smiles):
        molecule = _molecule_from_smiles(value)
        if use_morgan:
            assert fingerprint_generator is not None and fingerprints is not None
            fingerprint = fingerprint_generator.GetFingerprint(molecule)
            DataStructs.ConvertToNumpyArray(fingerprint, fingerprints[row_index])
        if use_descriptors:
            row: list[float] = []
            for _, function in descriptor_functions:
                try:
                    result = float(function(molecule))
                    row.append(result if np.isfinite(result) else np.nan)
                except Exception:  # RDKit descriptors can reject unusual valence states.
                    row.append(np.nan)
            descriptor_rows.append(row)

    names = [f"morgan_{index}" for index in range(n_bits)] if use_morgan else []
    if not use_descriptors:
        assert fingerprints is not None
        return FeatureSet(values=fingerprints, names=names)

    # Several legitimate size-dependent RDKit descriptors exceed float32 on
    # larger molecules. Keep descriptor precision and let fold-local model
    # pipelines handle missing values rather than converting them to infinity.
    descriptor_values = np.asarray(descriptor_rows, dtype=np.float64)
    descriptor_values[~np.isfinite(descriptor_values)] = np.nan
    descriptor_names = [name for name, _ in descriptor_functions]
    values = (
        np.hstack([fingerprints, descriptor_values])
        if fingerprints is not None
        else descriptor_values
    )
    return FeatureSet(values=values, names=names + descriptor_names)
