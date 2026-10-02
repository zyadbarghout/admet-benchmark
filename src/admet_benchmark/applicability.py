"""Original-fingerprint training references; no embedding distances or test fitting."""
from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def chemical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def scaffold(smiles):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles))


def fingerprints(smiles):
    generator = GetMorganGenerator(radius=2, fpSize=1024)
    return [generator.GetFingerprint(Chem.MolFromSmiles(chemical(s))) for s in smiles]


def training_neighbours(train_smiles, query_smiles):
    train, query = fingerprints(train_smiles), fingerprints(query_smiles)
    train_scaffolds = {scaffold(s) for s in train_smiles}
    if not train:
        raise ValueError('A nonempty training reference is required')
    nearest, similarity = [], []
    for fp in query:
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train))
        index = int(values.argmax()); nearest.append(index); similarity.append(float(values[index]))
    return {'nearest_train_index': nearest, 'nearest_train_similarity': similarity,
            'novel_scaffold': [int(scaffold(s) not in train_scaffolds) for s in query_smiles]}


def cliff_annotations(train_smiles, train_target, query_smiles, query_target, similarity=.8, difference=1.):
    """Post-evaluation stress cohort, never a training feature or selection rule."""
    train, query = fingerprints(train_smiles), fingerprints(query_smiles)
    flags, gaps = [], []
    for fp, target in zip(query, query_target, strict=True):
        similar = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train)) >= similarity
        gap = float(np.abs(np.asarray(train_target)[similar]-target).max()) if similar.any() else 0.
        flags.append(int(gap >= difference and similar.any())); gaps.append(gap)
    return {'property_cliff': flags, 'max_similar_property_gap': gaps}
