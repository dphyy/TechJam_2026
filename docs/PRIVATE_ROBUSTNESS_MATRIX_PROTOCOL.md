> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Private-Robustness Matrix Protocol

## Registered boundary

This matrix was generated before Phase 3 runtime implementation. It is local,
synthetic engineering evidence and is not organizer-private evidence. Selection
does not import Mercury, the evaluator, model scores, or simulator outcomes.

- Seed: `robustness-matrix-20260830-v1`
- Generator: `experiments/robustness_matrix_prepare.py`
- Lock directory: `artifacts/robustness-matrix-v1` (local and ignored)
- Frozen manifest SHA-256:
  `fda4900cfd8828ae8a861c51ecae3cbe7a2ef5f5ba7dacf81a19d65109b73727`
- Metamorphic fixture SHA-256:
  `f2384ecf4962e4861b349a66736c433fb34832115bf0ec7acb6387acb1bb7b6d`

The lock excludes every target and loose-title family in the released public
set and the seven consumed page-local provenance datasets. It chooses one
hash-selected member per remaining loose-title family, then assigns entire
deepest-category groups to training, screening, confirmation, or final. The
split unit therefore prevents target, loose-title family, and category-group
leakage.

## Frozen groups

| Split | Rows | Initial status |
|---|---:|---|
| Training | 480 | Open |
| Screening | 160 | Sealed |
| Confirmation | 80 | Sealed |
| Final | 80 | Sealed |

Every evaluation row has a separate annotation recording its split-specific
author, synthetic user, dialogue template, paraphrase family, unseen-wording
family, category group, hashed loose-title family, popularity band, and metadata
strata. The audit reports zero cross-split overlap for all seven grouping
dimensions and target IDs.

The screening, confirmation, and final splits each cover buying, browsing,
override, and boundary scenarios. They also cover low-popularity products,
missing price, short titles, sparse features, field contradictions, loose-title
near duplicates, and complete records. These labels describe source metadata;
they are never available to runtime code.

## Consumption discipline

`consumption-ledger.json` belongs to the immutable manifest by hash. Opening a
split requires a purpose, source commit, dataset hash, and optional configuration
hash. The command refuses to open confirmation before screening, or final before
confirmation:

```bash
python -m experiments.robustness_matrix_prepare \
  --output artifacts/robustness-matrix-v1 \
  --consume screening \
  --purpose "Phase 3 frozen admission comparison" \
  --config configs/PHASE3_CANDIDATE.json
```

The final split remains sealed through intermediate selection. Generated target
files and the mutable ledger remain ignored so target IDs do not enter source
history.

## Metamorphic suite

`data/metamorphic_robustness_v1.json` contains seven authored property groups:

- clause order, case, and punctuation;
- irrelevant filler;
- correction and override wording;
- negative and no-preference paraphrases;
- reordered alternatives;
- missing metadata remaining unknown;
- legal, unique output in every case.

The selected release passed legal output and candidate-membership checks but
failed four of seven complete case groups, specifically active-state equivalence
for ordinary reordering, correction wording, no-preference wording, and reordered
alternatives. This satisfies the detection gate without a target-specific
expected ranking and provides honest limitations for Phase 5.

## Reproduction

Pass the seven files under `artifacts/page-local-consumed-v1` as repeated
`--consumed-dataset` arguments when locking or verifying. Then run:

```bash
python -m experiments.metamorphic_validate \
  --config configs/selected.json \
  --output runs/metamorphic-selected-20260830.json
```

At freeze time the matrix had 47,786 eligible title families and 771 eligible
category groups after excluding 1,343 prior targets. The training selection
spans 116 whole category groups and screening spans 45; no single marketplace-
specific mega-category can consume a split's entire diversity budget. No
screening, confirmation, or final outcomes were opened at the Phase 3 source
freeze.
