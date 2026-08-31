> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Early paging with override reset protocol

Registered 30 August 2026 before running the candidate.

## Hypothesis

The first early-paging experiment failed because an intent-override message can
restate an already-active preference. In that case the ranking remains unchanged,
yet a target shown before the override may only become eligible on the override
turn. Paging then incorrectly moves away from ranks 1–10.

The new candidate starts paging on the first repeated ranking but forces page 1
whenever the ordinary runtime intent classifier reports `intent_override`. This
reset uses only the current message and conversation state; it does not read the
scenario type, target, evaluator state or eligibility gate. Ranking changes
continue to reset paging independently.

## Frozen comparison and gate

- Control: `configs/selected.json` (paging from turn 5)
- Candidate: `configs/paging_from_start_override_reset.json`
- Dataset: `data/public_set.jsonl`, all 200 consumed development sessions
- Evaluator: unchanged participant evaluator through `experiments.run`

Run a fresh control and candidate sequentially from the same source checkout.
Promote the candidate exactly when its overall TechnicalScore is greater than
or equal to the control, following the owner's stated rule. Report HitRate@10,
MRR, MTTC, scenario metrics, token usage, latency, fallbacks, agent errors and
source drift. Do not alter the override detector or paging condition after
observing results.

This second comparison is an explicit follow-up derived from the failed first
experiment. It is consumed public development evidence, not unseen validation.
