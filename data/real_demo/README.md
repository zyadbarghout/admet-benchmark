# Real-data demonstration snapshots

These local snapshots preserve the existing development/validation populations.
`provenance.json` records exact source and derived-file hashes. Original files
remain unchanged. They contain measured assay labels, not simulated labels.

- **KSOL:** 5,128 kinetic-solubility measurements from the preserved OpenADMET
  ExpansionRx development pool; evaluation uses log10 of positive source values.
- **LogD:** 4,934 measured distribution-coefficient values on molecules from that
  same KSOL development pool; evaluation uses the supplied LogD scale directly.
  This selection supports real overlapping-endpoint analysis; it is not claimed
  to be the entire original LogD dataset.
- **BBBP:** 1,631 original development/validation rows from the preserved
  MoleculeNet-derived classification snapshot. Registration collapses consistent
  replicates and quarantines conflicting identities, leaving 1,569 chemicals.

The `*_historical_structures.csv` files contain structures only, for overlap
exclusion. Historical test labels are absent from this example's training inputs.
See `../classification/README.md` for the preserved BBBP source and the original
loader's `OPENADMET_DATASET_ID` for the OpenADMET source.

No assay-unit conversion or salt/tautomer standardization is added. KSOL outputs
are described in log10 source assay units; LogD is already a logarithmic property.
Dataset ownership and license terms remain with their original providers. These
snapshots are for this local reproducible example; publication/distribution rights
have not been established by this implementation.
