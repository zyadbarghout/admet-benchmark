# Register a development study

The `admet-study` CLI prepares shared molecular partitions before any model is
fitted. Version 1 supports **repeated holdouts**, with one `fold=0` per repeat,
fixed model settings and either molecule or Bemis–Murcko scaffold grouping.
It does not run training, nested tuning or UQ calibration.

```bash
admet-study preflight configs/study_example.yaml
admet-study freeze configs/study_example.yaml --output outputs/my_study_v1
admet-study verify outputs/my_study_v1
```

Use a new output directory for every registration. `freeze` refuses existing
directories, even if they are empty. `verify` checks artifact hashes and rebuilds
assignments with the current runtime. Retain the manifest hash separately when
you need an external reference for integrity.

The example is author-created synthetic data with two overlapping endpoints.
Its units, model names, effects and CPU budget illustrate the schema only.

## Input contract

The CSV contains `sample_id,endpoint,smiles,observed`. Each row is a measured
chemical entity for one endpoint. Omit unmeasured endpoint rows; missing values
are rejected. Each declared endpoint must appear and undeclared endpoints fail.
Duplicate canonical chemistry within an endpoint is rejected rather than
silently averaged. Resolve replicate measurements under an explicit scientific
policy before registration. A sample ID may recur across endpoints only for
the same chemical entity.

Dataset paths resolve relative to the YAML. Declare the exact file SHA-256,
source and license. Registration checks the hash, not the legal validity of a
license declaration. Observations are stored on the submitted scale; endpoint
transforms are declared and domain-checked, not applied by this split producer.
Practical differences use the declared primary metric's evaluation scale.

## Identity and partition semantics

`rdkit_canonical_isomeric_v1` sanitizes and canonicalizes the submitted structure,
retaining its stereochemistry. It does not remove salts, normalize charge or
collapse tautomers. The frozen manifest records RDKit's version. A different
parent-structure policy requires another schema/policy version.

Molecule splits group by this identity. Scaffold splits group by RDKit's
Bemis–Murcko scaffold, with all acyclic molecules in one group. These groups
are shared across endpoints, preventing a compound from being training data
for one endpoint and evaluation data for another within the same repeat.

The test set uses `test_seed` once across all chemical groups. Every repeat then
partitions the remaining groups into training, calibration and validation.
`test_fraction` refers to all groups; `validation_fraction` and
`calibration_fraction` each refer to the remaining development groups.
Fractions count groups, not molecules, and are rounded up. The preflight audit
reports realized molecule and group support per endpoint/partition/repeat.

Assignment ordering uses SHA-256 of a stage, seed and chemical group. It is
independent of CSV row order and outcome values. Preflight rejects unsupported
class/molecule counts instead of repeatedly searching for a favorable split.
Different seeds can coincide for tiny datasets; the audit reports the number
of distinct assignments. Repeated holdouts share data and are not independent
experiments or k-fold CV.

## Registration bundle

| File | Purpose |
| --- | --- |
| `study.yaml` | Portable protocol pointing to the frozen observation snapshot |
| `observations.csv` | Retained submitted observations; its snapshot hash is in the portable protocol |
| `molecules.csv` | Submitted and canonical structures, chemical IDs and split groups |
| `fold_assignments.csv` | Endpoint/sample/chemical identity, group, repeat, fold, seed and partition |
| `preflight.json` | Original input hash, support audit, evidence label and warnings |
| `run_index.json` | Planned endpoint/model/repeat runs referencing the assignment hash |
| `manifest.json` | Artifact hashes, registration hash and runtime versions |

The original dataset hash remains in `preflight.json`; the sorted portable
snapshot has its own hash in `study.yaml`. The two files can have different
bytes while representing the same observations.

An external training adapter must consume these saved assignments, fit only
`train`, reserve `calibration` for UQ and emit predictions for the exact saved
validation/test contexts. That enforcement bridge and completed-run indexing
are not implemented yet. The existing model arena does not automatically
consume the new study files. Registration cannot certify that external code
obeyed the protocol or that test labels have never been inspected.
