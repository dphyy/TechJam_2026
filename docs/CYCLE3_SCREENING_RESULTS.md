> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Cycle 3 screening result: reranker admission

Recorded 27 August 2026. This is a completed fresh-screening comparison under the registration in [CYCLE3_EXPERIMENT_PROTOCOL.md](CYCLE3_EXPERIMENT_PROTOCOL.md). It is not organizer-private performance, a new public-set result, confirmation, or validation evidence.

## Lock and execution integrity

The local target lock was created before this source work with the committed generator and then verified read-only. It contains 160 screening sessions, excludes all 200 released-public targets and all 64 Cycle 2 target IDs under the documented loose-title heuristic, and has screening SHA-256 `f7dbd63005ca1b11346411a1592e351d33ea191302dd406fc3719f545b7b5624`. No final-validation target outcome was opened.

The valid control and both experimental runs used the same 160 screening bytes, local catalog, evaluator, selected model assets, 120-candidate ceiling, 30 reranked pairs, and 256-token pair limit. Each run's manifest reports `source_changed_during_run: false`, zero agent-error turns, and no startup fallback. The original pre-instrumentation C0 result was exactly equal on official outputs but did not have valid reranker-prefix telemetry; it is retained locally as an obsolete diagnostic run and was not used for selection.

| Mode | TechnicalScore | Hit@10 | MRR | MTTC | Prefix recall | Warm p95 | Prompt tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prefix` control | 0.811786 | 0.925000 | 0.635119 | 3.062500 | 0.937500 | 0.350050 s | 1,723,015 |
| `stratified` control | 0.799740 | 0.912500 | 0.621218 | 3.143750 | 0.918750 | 0.345959 s | 1,775,548 |
| `cover` candidate | 0.811942 | 0.925000 | 0.635640 | 3.062500 | 0.931250 | 0.425020 s | 1,723,496 |

Against `prefix`, `cover` has paired TechnicalScore delta `+0.000156` (exploratory 95% bootstrap interval `[0.000000, 0.000469]`, 10,000 resamples, existing runner seed `20260826`) and hit-rate delta `0.000000`. `stratified` has `-0.012045` TechnicalScore delta and `-0.012500` hit-rate delta. The fixed rank-spread control therefore falsifies the claim that any extra tail exposure helps.

## Decision

Neither candidate earns the registered `+0.010` screening gate. `stratified` is rejected for material regression. `cover` is retained only as a safe, disabled engineering experiment: its tiny MRR movement is not a practical gain and it lowered prefix recall. No admission configuration proceeds to confirmation or validation, no combination is authorized from these outputs, and the selected production configuration remains unchanged.

The next independent route was factored fielded retrieval, limited to the same fixed downstream budget. The fresh confirmation and validation splits remain unopened.

## Fielded retrieval result

The broad no-op control at source commit `2aa09ec` exactly reproduced the valid prefix control: `0.811786` TechnicalScore, `0.925000` Hit@10, `0.635119` MRR and `3.062500` MTTC. Its manifest also records no source drift.

| Mode | TechnicalScore | Hit@10 | Prefix recall | Candidate-pool recall | Warm p95 |
|---|---:|---:|---:|---:|---:|
| `broad` control | 0.811786 | 0.925000 | 0.937500 | 0.968750 | 0.350816 s |
| `field_union` simple control | 0.741564 | 0.837500 | 0.850000 | 0.975000 | 0.826006 s |
| `factored` candidate | 0.649234 | 0.731250 | 0.737500 | 0.843750 | 0.282576 s |

Against `broad`, the union's paired TechnicalScore delta is `-0.070222` (exploratory 95% bootstrap interval `[-0.111618, -0.031724]`) and the strict fielded route's is `-0.162552` (`[-0.217374, -0.109737]`), both with 10,000 resamples and seed `20260826`. No agent error, fallback, source drift, confirmation, or validation occurred.

Both variants are rejected and disabled. The slight union candidate-pool-recall increase is not useful because it displaces high-quality reranker-prefix candidates; a larger pool count is not an end-to-end gain. There will be no fielded-retrieval combination or threshold sweep. The next independent route is equal-budget, source-grounded reranker serialization; confirmation and validation remain unopened.

## Grounded reranker serialization result

The no-op `head` control at source commit `e54bf83` again reproduced the prior official output exactly: `0.811786` TechnicalScore, `0.925000` Hit@10, `0.635119` MRR and `3.062500` MTTC. All three document modes retained 93.75% prefix recall and had no source drift, fallbacks, or agent errors.

| Mode | TechnicalScore | Hit@10 | MRR | MTTC | Prompt tokens | Warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| `head` control | 0.811786 | 0.925000 | 0.635119 | 3.062500 | 1,723,015 | 1.455258 s |
| `lexical` control | 0.803141 | 0.918750 | 0.620052 | 3.112500 | 1,298,611 | 0.473824 s |
| `protected` candidate | 0.797647 | 0.912500 | 0.615491 | 3.162500 | 1,082,712 | 0.342561 s |

Against `head`, lexical has paired TechnicalScore delta `-0.008645` (exploratory 95% bootstrap interval `[-0.028169, 0.010074]`) and protected has `-0.014138` (`[-0.031513, 0.001192]`), with 10,000 resamples and seed `20260826`. The head-control p95 was anomalously high compared with prior clean local runs, so it is not a reliable standalone resource claim; this does not affect the deterministic official-score comparison.

Both modes are rejected. Lower token use does not compensate for lower Hit@10 and MRR, and no serializer is authorized for confirmation or validation. With all three primary families failing their predeclared gate, the campaign now returns to an evaluation-only failure-cluster audit before registering any new mechanism. The fresh confirmation and validation splits remain unopened.
