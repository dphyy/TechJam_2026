> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Continuation roadmap release report

All registered continuation phases have now been implemented and evaluated. No
candidate is promoted, because the only screening survivor failed the frozen
confirmation comparison. `configs/selected.json` remains unchanged and is the
release configuration.

## Final phase disposition

| Work | Screening outcome | Final disposition |
|---|---|---|
| Phase 7 fresh matrix and attribution | Infrastructure and provenance passed | Retained |
| Phase 8 canonical state semantics | Correctness invariant passed; small aggregate cost | Opt-in foundation only |
| Phase 9 catalog vocabulary v2 | Token cap failed | Rejected |
| Phase 10 admission v2 | HitRate preserved; TechnicalScore +0.000385 vs selected | Advanced to confirmation |
| Phase 4 adaptive D20/D30 | Pair target passed; latency and boundary gates failed | Rejected |
| Phase 11 domain MiniLM | Open ranking improved; screening resource/slice gates failed | Rejected |
| Phase 12A rejection continuity | Lost one buying hit | Rejected |
| Phase 12B typed question | MRR, MTTC, productivity, and latency regressed | Rejected |

## Confirmation decision

The confirmation split hash was
`0811fac13c886751af72364e45066f37bd81394424b54a89fbd099cdfd8766a4`.
It was opened once after source and configuration freeze. Both runs preserved
HitRate at `0.9875` and had no fallback or agent-error turns.

| Metric | Selected control | Admission v2 | Delta |
|---|---:|---:|---:|
| HitRate@10 | 0.987500 | 0.987500 | 0.000000 |
| MRR | 0.664990 | 0.645312 | -0.019678 |
| MTTC | 2.125000 | 2.012500 | -0.112500 |
| TechnicalScore | 0.870747 | 0.867094 | -0.003653 |
| Warm p95 | 0.367791 s | 0.540863 s | +47.06% |
| Prompt tokens | 778,239 | 731,329 | -6.03% |

The paired attribution contains no gained or lost hits: 79 sessions remain
successful and the same browsing session remains an admission miss. The lower
candidate score comes from ordering and runtime, not retrieval or correctness.
It therefore fails the default promotion gate and is not written into
`configs/selected.json`.

No arm independently passed confirmation, so there is no valid combination arm
and no reason to open the remaining original final evidence. Further tuning
requires a newly frozen, source-disjoint matrix; the v2 screening and
confirmation sets are now consumed evidence.

## Verification

The final tree passes 622 unit tests, Ruff, and dependency checks. Candidate
features remain default-off, model assets are hash-verified and offline-only,
and missing or corrupt optional assets preserve the explicit sparse/BM25
fallback. Local run traces, generated matrices, and trained weights remain
ignored artifacts with committed reproducibility scripts and reports.
