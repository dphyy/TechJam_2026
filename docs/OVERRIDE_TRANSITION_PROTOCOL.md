> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Override transition protocol

This change is accepted only if all of the following hold without revising the
rules after seeing benchmark outcomes:

- State-transition capability tests pass for replacements, polarity changes,
  category changes, alternative narrowing, and attribute-only retractions.
- Additive preferences and protected no-change/either-or language do not
  produce override decisions.
- Explicit correction language still detects a zero-delta restatement, because
  paging must return to page 1 even when the requested ranking is unchanged.
- The selected public evaluation has no agent errors and its Technical Score is
  at least the current selected reference, `0.839390` (allowing only display
  rounding in the recorded reference).
- Hit Rate@10 and the intent-override slice do not regress.

The implementation may be rejected as a unit if any gate fails. Benchmark
results are evidence for acceptance only; they are not used to add phrases or
tune thresholds.

## Recorded result

Accepted on 30 August 2026. The 200-session public evaluation reproduced the
reference exactly: Technical Score `0.839390`, Hit Rate@10 `0.970000`, and
intent-override Hit Rate@10 `0.900000` with MRR `0.748611`. The run reported
zero agent-error turns and zero fallbacks. The private-like engineering pack
passed 19/19 assertions, and the full project suite passed 547 tests.
