import pandas as pd

from admet_benchmark.chemistry import ChemicalFilter, apply_chemical_filter


def test_substitution_filter_uses_external_heavy_atom_attachments():
    frame = pd.DataFrame(
        {
            "smiles": ["c1ccccc1", "Cc1ccccc1", "CCO"],
            "identifier": ["benzene", "toluene", "ethanol"],
        }
    )
    retained, audit = apply_chemical_filter(
        frame, "smiles", ChemicalFilter("c1ccccc1", "substituted")
    )

    assert retained["identifier"].tolist() == ["toluene"]
    assert audit.loc[0, "chemistry_substitution_state"] == "unsubstituted"
    assert audit.loc[1, "chemistry_external_heavy_atoms"] == 1
    assert audit.loc[2, "chemistry_substitution_state"] == "not_matched"
