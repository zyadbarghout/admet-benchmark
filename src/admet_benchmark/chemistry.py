"""SMARTS-based filtering with explicit substitution annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from rdkit import Chem


SubstitutionState = Literal["any", "substituted", "unsubstituted"]


@dataclass(frozen=True)
class ChemicalFilter:
    """A substructure query and the required external attachment state.

    A match is called substituted when at least one atom in the SMARTS match has
    a heavy-atom neighbour outside the matched atom set. This is deliberately
    reported as an external attachment, because fused ring systems also meet
    this definition.
    """

    smarts: str
    substitution: SubstitutionState = "any"

    def __post_init__(self) -> None:
        if not self.smarts or Chem.MolFromSmarts(self.smarts) is None:
            raise ValueError(f"Invalid SMARTS query: {self.smarts!r}")
        if self.substitution not in {"any", "substituted", "unsubstituted"}:
            raise ValueError(
                "substitution must be 'any', 'substituted', or 'unsubstituted'"
            )


def _match_annotation(molecule: Chem.Mol, query: Chem.Mol) -> tuple[int, int]:
    """Return the number of SMARTS matches and unique external heavy atoms."""

    matches = molecule.GetSubstructMatches(query)
    external_atoms: set[int] = set()
    for match in matches:
        matched_atoms = set(match)
        for atom_index in match:
            atom = molecule.GetAtomWithIdx(atom_index)
            for neighbour in atom.GetNeighbors():
                if (
                    neighbour.GetIdx() not in matched_atoms
                    and neighbour.GetAtomicNum() > 1
                ):
                    external_atoms.add(neighbour.GetIdx())
    return len(matches), len(external_atoms)


def annotate_chemical_filter(
    frame: pd.DataFrame,
    smiles_column: str,
    chemical_filter: ChemicalFilter,
) -> pd.DataFrame:
    """Annotate each row with SMARTS-match and external-attachment evidence."""

    if smiles_column not in frame:
        raise KeyError(f"Missing SMILES column: {smiles_column}")

    query = Chem.MolFromSmarts(chemical_filter.smarts)
    annotations: list[dict[str, object]] = []
    for smiles in frame[smiles_column].fillna(""):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            annotations.append(
                {
                    "chemistry_valid_smiles": False,
                    "chemistry_match_count": 0,
                    "chemistry_external_heavy_atoms": 0,
                    "chemistry_substitution_state": "invalid_smiles",
                    "chemistry_passes_filter": False,
                }
            )
            continue

        match_count, external_count = _match_annotation(molecule, query)
        if match_count == 0:
            state = "not_matched"
        elif external_count > 0:
            state = "substituted"
        else:
            state = "unsubstituted"

        passes = match_count > 0 and (
            chemical_filter.substitution == "any"
            or state == chemical_filter.substitution
        )
        annotations.append(
            {
                "chemistry_valid_smiles": True,
                "chemistry_match_count": match_count,
                "chemistry_external_heavy_atoms": external_count,
                "chemistry_substitution_state": state,
                "chemistry_passes_filter": passes,
            }
        )

    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(annotations)], axis=1
    )


def apply_chemical_filter(
    frame: pd.DataFrame,
    smiles_column: str,
    chemical_filter: ChemicalFilter | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return retained rows plus an audit table for every input molecule."""

    if chemical_filter is None:
        audit = frame.copy()
        audit["chemistry_passes_filter"] = True
        return frame.reset_index(drop=True), audit

    audit = annotate_chemical_filter(frame, smiles_column, chemical_filter)
    retained = audit.loc[audit["chemistry_passes_filter"]].reset_index(drop=True)
    return retained, audit
