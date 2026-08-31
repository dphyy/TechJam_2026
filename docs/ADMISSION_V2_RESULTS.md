> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Admission v2 results

Phase 10 keeps the frozen v1 coefficients while moving catalog tokenization,
field presence, product type, and metadata completeness out of the turn path.
Catalog tokens are represented as sorted 32-bit integer arrays. Unknown query
tokens are never inserted into the catalog vocabulary, so they cannot create a
false overlap. A missing or invalid model still falls back to the original BM25
prefix.

The frozen open benchmark used 80 Phase 7 training targets. V2 matched v1 at
Recall@20 and Recall@30 (`0.9875` each). Its conditional admission MRR was
`0.937398`, compared with `0.939481` for v1, and its mean Top-30 membership
overlap with v1 was `0.963333`. Feature extraction p95 fell from `69.08 ms` to
`2.19 ms`, below the registered `5 ms` limit. The model is provenance-bound to
the catalog and v1 parent hash in `models/admission_linear_v2.json`.

The source-matched screening run is
`runs/phase10-screening-compact-candidate-20260830`. Against the selected Phase
8 control, HitRate remained `0.98125`, TechnicalScore increased from `0.864686`
to `0.865071`, and the rerank-prefix recall increased from `0.95625` to
`0.98125`. Warm p95 was `0.424832 s`, below the `0.534 s` cap. Neural pairs
remained D30, tokens declined from `1,512,580` to `1,509,496`, there were no
fallbacks, and RSS was `1,190,182,912` bytes versus `1,179,467,776` bytes for
the selected control.

The candidate passes the Phase 10 gate. It remains a candidate rather than
silently replacing `configs/selected.json`; adaptive D20/D30 and later ranking
or policy arms must still pass their own frozen comparisons.
