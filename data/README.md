# Competition data

Data preparation for the current **lexical search with guarded paging** pipeline.
For execution and current results, see the [project README](../README.md).

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions. This is consumed development evidence, not an untouched holdout.

Expected SHA-256:
`857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

Obtain `catalog.jsonl.gz` and its published archive checksum from the organizer's
participant release. Verify the archive before decompressing it as
`data/catalog.jsonl`; the catalog is not bundled in this source checkout.
Expected row count: 50,000. Expected decompressed SHA-256:
`da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.

From the repository root:

```bash
gzip -dk data/catalog.jsonl.gz
shasum -a 256 data/catalog.jsonl data/public_set.jsonl
```

Keep the original data and checksums unchanged. No embeddings, model weights or
prebuilt index are required for the current default. See [setup](../docs/SETUP.md)
and [data attribution](../DATA_ATTRIBUTION.md).

Never place API keys, private evaluation data, or participant outputs in this directory.

Dataset filenames such as `final-sealed.jsonl` and `reserved.jsonl` describe
historical split roles, not current exposure. Before claiming a set is untouched,
consult [dataset status semantics and the receipt audit](../docs/DATASET_STATUS.md).
The robustness-v1 screening, confirmation and final sets have recorded
consumption; changing source code does not make them fresh evaluation evidence.
