> **Historical development record — not current release guidance.** Retained to explain implementation progress and earlier experiment decisions. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Main-to-generalization merge decisions

Recorded 28 August 2026 after merging `main` into `feat/improve-generalization`.

## Release boundary

- **Submission and reliability:** keep `configs/selected.json` on the 30-candidate MiniLM rerank prefix. This is the lower-cost architecture with the established release record.
- **Public-score demonstration only:** D120 measured `0.807170` on the already-used public development set. It failed fresh screening and the registered latency gate, so it is an experiment, not the selected release or evidence of private performance.
- **Potential future candidate:** D60 remains eligible only for a newly registered comparison on unseen sessions. Its public result and earlier fresh-screen delta do not authorize promotion.

The merged runtime supports Cycle 3 admission, retrieval, document, and depth arms alongside the intent/planning experiments. All score-regressing behavior remains gated. The selected release still reranks 30 candidates.

## Dialogue decision

Named attribute questions are single-use per session. The `other` API facet may still be used within its bounded limit because additional turns reveal open-vocabulary intent, but the visible prompt is selected from a non-repeating sequence. An unproductive reply stops further generic questioning. A strict one-`other` experiment scored `0.746536` on the public development set and was rejected because it materially reduced target recovery and dialogue efficiency.

Intent remains an explicit, target-independent `buying`, `browsing`, or `mixed` decision. The typed retrieval plan carries positive and negative terms, hard and soft requirements, use case, alternatives, provenance, and optional component scope. Intent routing and intent-aware questioning remain separately configurable because their earlier public runs did not pass promotion. Future work should improve and test the retrieval/dialogue decision jointly on unseen sessions rather than silently enabling the rejected arms.

## Price decision

Budget assertions are stored as soft preferences, including emphatic wording. Catalog price is too incomplete and inconsistent to justify hard filtering.

- Exact finite prices inside a requested range receive a small positive adjustment.
- Exact finite prices outside the range receive a small negative adjustment.
- Missing, malformed, and inconclusive lower-bound prices remain neutral.
- No price condition removes a candidate or enters the hard contradiction groups.

The adjustment is bounded by `soft_price_weight` and defaults to `0.02`. Diagnostics expose per-product `price_adjustments`.

## Evidence policy

Historical scores stay attached to their measured source/configuration. Any post-merge score must be recorded as a new measurement. Public development results must not be relabeled as unseen validation, and neither D60 nor D120 may be promoted without the declared evidence above.

The merged selected configuration was remeasured on the 200-row public development set after the varied-question and soft-price changes:

| HitRate@10 | MRR | MTTC | TechnicalScore | Prompt tokens |
|---:|---:|---:|---:|---:|
| 0.895000 | 0.613746 | 3.245000 | 0.786724 | 2,375,167 |

This exactly matches the established official metrics. It is a reproducibility/development check, not new validation evidence. A deliberately strict one-`other` run scored `0.746536` and remains a recorded negative result rather than the selected dialogue policy.

Post-merge verification: 419 tests, Ruff, `pip check`, and the 19-assertion private-like pack all pass; the private-like run recorded zero failed/unverified assertions, API errors, and fallback turns.
