> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Margin-aware reranking protocol

Registered 30 August 2026 before running either arm on the Cycle 5
popularity-matched target pack. This is one bounded ranking hypothesis, not a
license to tune thresholds on score-bearing outcomes.

## Hypothesis

The selected ranker always gives the MiniLM ordering weight `0.75`. Earlier
evidence shows that larger or more semantic rerankers can recover additional
targets while lowering MRR. A small top-logit margin is target-independent
evidence that the ranker's first choice is uncertain. In that case only, retain
more of the original lexical ordering by reducing neural weight to `0.50`.

The candidate changes no retrieval route, candidate budget, model, question,
slate, state, or fallback behavior. It uses the already recorded raw logit
margin. The threshold `1.00` is fixed from the existing grouped margin
calibration boundary; it is not selected from Cycle 5 targets.

## Frozen arms

- Control: `configs/selected.json`
- Candidate: `configs/margin_fusion.json`
- Screening data: `artifacts/cycle5/synthetic-targets/screening.jsonl`
- Confirmation data: `artifacts/cycle5/synthetic-targets/confirmation.jsonl`
- Validation data: `artifacts/cycle5/synthetic-targets/validation.jsonl`

Run the control and candidate once on screening. Continue only if the candidate:

1. improves TechnicalScore by at least `0.005`;
2. does not lower HitRate@10 or MRR;
3. records zero API errors, fallbacks, or source drift; and
4. stays within 10% of control p95 response latency and uses the same model-pair
   count.

If screening passes, freeze source and run confirmation once. Promote only if
confirmation has a positive TechnicalScore delta, no HitRate@10 loss, no MRR
loss, and the same operational checks. Validation is a final one-shot
verification after selection, never a tuning set. A failed arm remains gated and
its result is recorded.

All Cycle 5 sessions are synthetic catalog-target recovery under the unchanged
participant evaluator. They are more representative than uniform catalog draws,
but they are not organizer-private performance evidence.
