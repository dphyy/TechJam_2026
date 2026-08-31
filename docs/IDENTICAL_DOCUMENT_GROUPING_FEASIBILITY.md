> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Exact-document grouping feasibility

Evaluated on 30 August 2026 before Phase 1B runtime implementation.

Phase 1B proposed scoring one representative when multiple candidates in the
same MiniLM call have byte-identical serialized documents. A source-matched
upper-bound audit grouped the exact `head` documents for every selected D30 model
call. Cache-hit turns were excluded because they do not invoke the reranker.

| Workload | Model calls | Candidate pairs | Unique documents | Maximum saved pairs | Maximum reduction |
|---|---:|---:|---:|---:|---:|
| Public development | 457 | 13,710 | 13,699 | 11 | 0.0802% |
| Unseen development | 182 | 5,460 | 5,458 | 2 | 0.0366% |

These are optimistic ceilings: they assume grouping itself costs nothing. Both
are far below the roadmap's 20% evaluated-pair gate, and adding a synthetic
duplicate-heavy workload would not demonstrate useful submission behavior.

No Phase 1B runtime flag or grouping branch was added. Distinct IDs and documents
continue through the selected reranker unchanged. This is a measured feasibility
rejection, not a claim that the catalog contains no duplicate products.
