> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Cycle 4 slate-paging registration

Registered 28 August 2026, before any candidate run. It follows the recorded
finding that the agent re-serves a byte-identical slate for a mean of seven turns
in every failed released-public session, and that seventeen of twenty-one missed
targets already sit inside the computed but never displayed positions 11-120.
See [the finding](CYCLE4_SLATE_REPETITION_FINDING.md).

## Single candidate

`cycle4_paging` changes exactly one configuration field,
`slate_paging_first_turn`, from `0` (disabled) to `5`. Retrieval, the candidate
ceiling, reranking depth and model, blend weight, state, question policy,
document serializer, slate size and evaluator are unchanged. It uses no target,
sample, or label information at runtime, and adds no retrieval, inference,
memory, or latency cost: the paged positions are already computed and discarded
by the current build.

When a turn's ranking is identical to the previous turn's and the turn number is
at least five, the agent serves the next unseen page of its own ranking instead
of repeating the previous slate. Any change in the ranking resets to the top
slate immediately.

## Rationale for the two conditions

Re-serving an identical slate has an exactly zero probability of scoring: the
session ends the moment the target enters the scored slate, so a still-running
session proves the target is absent from the slate just shown, and an unchanged
ranking reproduces that slate exactly.

The turn-5 threshold follows the published rule that the preference change in an
Intent Override session lands on turn 3 or 4, which the official simulator
implements as `rng.choice([3, 4])`. Before the override is delivered a hit is not
scorable, so a target already displayed is not yet counted; paging away from it
early would discard a session the current build wins. From turn 5 no gate can
still be closed.

## Pre-registered decision boundary

Evaluate once on the Cycle 3 screening split against the recorded 30-depth C0
screening control, after reproducing that control on the measuring machine. The
candidate earns confirmation only with all of:

- at least `+0.010` screening TechnicalScore against C0;
- no more than `-0.010` screening Hit@10 against C0;
- valid responses on every turn, with no fallback, agent error, or source drift;
- warm p95 latency at most the matched control's p95 plus 10 percent, since the
  change performs no additional work.

## Released-public expectation

An exact counterfactual replay of the recorded baseline traces predicts
`0.839176` TechnicalScore and `0.9700` Hit@10 on the released public 200, from
`0.786724` and `0.8950`. The replay reproduces all four official baseline metrics
to six decimal places, and the simulator's replies depend only on
`ask_attribute`, never on the displayed recommendations, so the prediction is
exact rather than estimated for that consumed dataset.

A public run therefore serves as an implementation check: a released-public
result materially different from `0.839176` indicates an implementation defect,
not a modelling result. It remains descriptive and cannot satisfy any gate.

## Scope limits

No further paging variants, thresholds, anchored-head designs, or page sizes
follow from this arm regardless of outcome. The variant comparison recorded in
the finding used consumed data, and that selection pressure is disclosed.
Confirmation and validation packs remain unopened.
