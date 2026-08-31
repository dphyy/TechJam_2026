> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Cycle 3C full-pool reranking registration

Registered 27 August 2026 after the completed D60 result. D60 increased public Hit@10 and moved the released score to `0.797064`, but it failed fresh-screen promotion and remains unselected. The existing fresh confirmation and validation packs remain unopened.

## Single candidate

D120 reranks every member of the existing 120-candidate pool instead of a 30- or 60-item prefix. This is the maximal bounded depth already supplied by the selected sparse retrieval; it is a single full-pool feasibility test, not a 90/120 depth sweep. The catalog, 120-candidate ceiling, local model revision, blend weight, retrieval, state, question policy, document serializer, output size, and evaluator remain unchanged. It uses no target, sample, or label information at runtime.

The public failure audit found that every released-public miss entered this 120-item pool. D60 made the expected prefix-recall change but did not reach its fresh score gate. D120 tests whether the remaining tail can be ranked without inventing a new retrieval mechanism. Its additional compute cost is a liability that must be disclosed, not a reason to relax measurement.

## Evaluation and decision boundary

Run D120 once on the already-consumed Cycle 3 screening split using the committed one-field configuration. Compare it with the recorded 30-depth C0 screening result. Record hashes, official metrics, prefix recall, tokens, latency, RSS, errors, and fallbacks. It earns confirmation only with at least `+0.010` screening TechnicalScore, no more than `-0.010` Hit@10, valid responses, no fallback/error, and p95 no more than twice the matched C0 clean serial run. There are no further depth variants or combinations regardless of outcome.

Because the owner explicitly set a released-public score objective, run D120 exactly once on the already-consumed public set after the screening run is complete. That result is descriptive only even if it crosses `0.80`: it cannot repair a failed screening gate, consume confirmation/validation, or be relabeled as an independent result. Report the result and resource cost without rerunning or tuning D120.
