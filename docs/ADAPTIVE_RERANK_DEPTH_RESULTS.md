> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Adaptive rerank depth results

After Phase 10 passed, the previously skipped D20/D30 experiment was
re-registered with one fixed monotonic rule. The candidate chooses D20 when the
normalized admission-score separation between ranks 20 and 30 is at least
`0.02`; otherwise it retains D30. The rule never uses target IDs, evaluator
labels, phrases, or future turns, and exposes its selected depth, gap, and reason
in diagnostics.

On the open 80-target admission check, the frozen rule selected D20 for 67
queries, reduced the theoretical pair count by `27.92%`, and preserved the v2
target recall of `0.9875`.

The one-shot screening run is
`runs/phase4-adaptive-depth-screening-20260830`. It preserved HitRate at
`0.98125`, increased TechnicalScore from `0.865071` to `0.867535`, and reduced
neural pairs from `10,020` to `7,310` (`27.05%`). However, warm p95 improved
only from `0.424832 s` to `0.391378 s` (`7.87%`), below the registered 20%
requirement, and boundary MRR fell from `0.483503` to `0.453146`.

The arm is rejected. `configs/adaptive_rerank_depth.json` remains an auditable
opt-in experiment; D30 remains the candidate release budget and confirmation
was not opened for this arm.
