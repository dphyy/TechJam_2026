> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Early-paging comparison protocol

Registered 30 August 2026 before running either arm.

## Hypothesis

The selected release waits until turn 5 before advancing through an unchanged
ranking. That protects a target which appears in the first slate but is not yet
eligible under an intent-override gate. In practice, an actual intent override
should change the full ranking and `_slate_page` already resets to page 1 on any
ranking change. Advancing from the first repeated ranking may therefore expose
more candidates earlier without harming intent-override recovery.

`slate_paging_first_turn: 1` means the first response still returns ranks 1–10;
the second response returns ranks 11–20 only when the entire ranking is
unchanged. A changed ranking always resets to ranks 1–10.

## Frozen comparison

- Control: `configs/selected.json` (`slate_paging_first_turn: 5`)
- Candidate: `configs/paging_from_start.json` (`slate_paging_first_turn: 1`)
- Dataset: `data/public_set.jsonl`, all 200 consumed development sessions
- Evaluator: unchanged participant evaluator through `experiments.run`

Run a fresh control and candidate sequentially from the same source checkout.
Promote the candidate exactly when its overall TechnicalScore is greater than
or equal to the control TechnicalScore, as requested by the project owner.
Report HitRate@10, MRR, MTTC, per-scenario metrics, fallbacks, API errors and
source drift as secondary evidence. Do not tune a different start turn after
observing the result.

This is public development evidence, not an unseen or organizer-private result.
