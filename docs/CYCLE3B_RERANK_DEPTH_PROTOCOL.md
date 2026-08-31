> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Cycle 3B reranker-depth registration

Registered 27 August 2026 after the completed Cycle 3 admission, fielded-retrieval, and document-serialization screening results. Those three mechanism families failed their fixed screening gates and remain disabled. Confirmation and validation targets remain unopened.

## Why this is a new, bounded branch

An evaluation-only audit of the selected public run found 21 misses. Every missed target appeared in the 120-item ranked pool at least once: four had best rank 11–30 and seventeen had best rank 31–120. The prior candidate-admission experiments did not recover this tail without harming prefix quality. Therefore the only new candidate is a larger *existing* neural prefix: 60 rather than 30 candidates. No model, model weight, retrieval route, state parser, question policy, Top 10 limit, source serializer, or target-aware behavior changes.

This is a feasibility tradeoff, not a claim of algorithmic novelty. A historical pre-Cycle-2 run directionally improved with 60 candidates but was not selected because it approximately doubled neural work and its old held-out comparison was inconclusive. It is not evidence for this source or configuration and is not reused as a holdout result.

## Fixed comparison

| Arm | `rerank_limit` | All other selected settings |
|---|---:|---|
| C0 | 30 | unchanged |
| D60 | 60 | byte-identical selected configuration except this integer |

Run C0 and D60 once each on the existing Cycle 3 screening split after the configuration is committed. Record official metrics, exact run/source/config/input hashes, candidate/prefix recall, latency, peak RSS, tokens, errors, and fallbacks. The screening data is development data; it may select or reject this one declared candidate, but it is not a new holdout.

D60 earns confirmation only if it gains at least `+0.010` TechnicalScore against C0 on screening, does not lose Hit@10 by more than `0.010`, has no correctness/API/fallback failure, and has warm p95 no more than twice C0 in a clean serial measurement. The earlier fixed one-second local target is reported separately; an externally busy host makes that absolute figure unverified rather than a pass. There are no depth, weight, or combination sweeps after this run.

If D60 earns confirmation, freeze source/config/model/input hashes, then run C0 and D60 exactly once each on the unopened Cycle 3 confirmation split. Promotion to final validation requires non-negative confirmation score delta, no more than `-0.010` Hit@10 delta, and the same correctness/resource gates. Freeze the finalist before a one-time validation comparison. Any source or configuration repair after either gate consumes that split for the changed source.

Only after fresh-screening selection may the released 200-session set be run as a descriptive regression check. A public score above `0.80` would still be development evidence, not organizer-private performance, a prize prediction, or a reason to rewrite failed arms.

## Registered public descriptive measurement

D60 completed screening with `+0.001705` TechnicalScore, unchanged Hit@10, higher prefix recall, and p95 within twice C0. It fails the `+0.010` promotion gate; it cannot run confirmation or validation and cannot become the selected configuration from any public result.

The owner separately set a released-public score target. To answer that narrow question without reopening selection, run D60 exactly once on the already-consumed 200 released sessions after this amendment is committed. Compare it with the recorded selected public C0 score of `0.786724`. This one run is descriptive only: do not alter D60, rerun it, combine it, relabel it as fresh evidence, or use it to override the failed screening gate. Report the exact score and resource result whether it improves or regresses.
