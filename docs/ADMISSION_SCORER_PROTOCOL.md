# All-Pool Admission Scorer Protocol

## Pre-registered hypothesis

A cheap target-independent scorer over all 120 retained BM25 candidates can
improve admission Recall@20 and Recall@30 before MiniLM, without weakening
deterministic guards or adding more than 10% to warm p95 latency.

The comparison order is fixed:

1. selected BM25 prefix;
2. inspectable deterministic feature fusion;
3. L2-regularized linear scoring only if fusion leaves measurable headroom.

Screening, confirmation, and final outcomes were unopened while implementing
and fitting these arms. Candidate source, configuration, and model are committed
before screening is consumed.

## Runtime boundary

`admission-features-v1` reads only:

- normalized BM25/fused score, original rank, and route agreement;
- query-term coverage in title, category, feature, detail, description, and all
  ordinary fields;
- target-independent object/accessory compatibility;
- positive, negative, hard, and budget evidence from the active ledger;
- catalog metadata completeness.

Missing price or an absent field contributes neutral evidence. Proven conflicts
may lower compatibility; absence alone cannot. The scorer neither changes the
candidate set nor bypasses the existing constraint and legality guards.

The linear model is `models/admission_linear_v1.json`, SHA-256
`3754eafd076e70e377120fc20e26de4041df2900a7e63f3bf7515819da9b4b82`.
It pins the feature order/version, preprocessing means and scales, coefficients,
intercept, seed, algorithm, catalog hash, and training/validation evidence
hashes. A missing, malformed, or catalog-mismatched model records a fallback and
uses the selected BM25 prefix.

## Training evidence

Training uses only the matrix's open 480-target split. Three deterministic
catalog-derived query families combine title tails, deepest category words,
structured facets, and feature words. The retained BM25 neighbors are the hard
negatives. Loose-title families are hash-grouped into fit and internal validation
folds.

The fit contains 994 queries / 119,280 pairs. Internal validation contains 317
queries / 38,040 pairs; 124 generated query variants whose target did not enter
the 120-candidate pool are recorded as retrieval misses and cannot be rescued by
admission.

| Admission arm | Recall@20 | Recall@30 | Conditional MRR |
|---|---:|---:|---:|
| BM25 prefix | 0.829653 | 0.876972 | 0.541486 |
| Deterministic fusion | 0.858044 | 0.892744 | 0.569525 |
| Regularized linear | 0.880126 | 0.914826 | 0.575270 |

The linear arm has the best aggregate results and does not regress Recall@30 on
complete, contradictory-field, missing-price, near-duplicate, short-title, or
sparse-feature slices. Its Brier score is `0.065238`; pure-Python model scoring
measured `1.256` microseconds per candidate over the validation matrix. These are
training-proxy results, not promotion evidence.

## Screening and promotion gates

- Recall@20 should approach or exceed selected Recall@30 on more than one fresh
  split; Recall@30 must not regress overall or on critical slices.
- End-to-end screening and confirmation must have no HitRate@10 loss and no
  unexpected fallback.
- All-pool scoring may add no more than 10% to selected warm p95.
- Deterministic, metamorphic, legality, and failure tests must remain clean.
- Confirmation opens only if the committed linear candidate passes screening.
- `configs/selected.json` remains unchanged until both fresh gates pass.

## Frozen screening result

Screening opened once at source commit `41a4cb6` against dataset SHA-256
`9d470a74b28231d680ae7d0d71fd75751792ff9b051f8e773df559951a8331ba`.
There were no startup or turn fallbacks.

| Arm | HitRate@10 | MRR | MTTC | TechnicalScore | p95 |
|---|---:|---:|---:|---:|---:|
| Selected BM25 prefix | 0.956250 | 0.558435 | 3.337500 | 0.798906 | 0.356s |
| Deterministic fusion | 0.962500 | 0.563279 | 3.300000 | 0.804234 | 0.586s |
| Regularized linear | 0.968750 | 0.588906 | 3.056250 | 0.819922 | 0.590s |

Both all-pool arms improved overall technical quality. Fusion preserved or
improved scenario HitRate, while linear lost one browsing hit despite stronger
overall results. However, fusion increased p95 by about 64.6% and linear by
about 65.7%, both far beyond the registered 10% ceiling. The repeated cost in
full catalog-field and preference-evidence extraction dominates the inexpensive
linear arithmetic.

Phase 3 is rejected on its resource gate. Confirmation was not opened, the
candidate configurations remain diagnostic only, and `configs/selected.json`
is unchanged. Because Phase 4 requires a passing calibrated Phase 3 scorer,
adaptive D20/D30 evaluation is not authorized from this branch of evidence.
