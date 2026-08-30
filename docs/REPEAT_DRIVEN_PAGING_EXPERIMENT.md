# Repeat-driven paging experiment

Registered 30 August 2026 before generating the new target pack or running
either arm. This experiment follows the product-logic objection that an exactly
repeated Top-10 slate cannot produce a new target hit in the official simulator.

## Arms

- Control: the locked finalist in `configs/selected.json`, which first advances
  a stable Top-10 on turn 3.
- Candidate: `configs/exp06_repeat_driven_paging.json`, which first advances a
  stable Top-10 on turn 2, the earliest possible repeated turn.

Both arms use identical source, models, retrieval, ranking, limits, and
questions. The only configuration difference is `slate_paging_first_turn`.
The existing policy compares Top-10 membership without regard to order. A
changed member or semantic override resets to the current head; an unchanged
head selects the highest-ranked products not previously shown.

## Fresh evidence

Generate a new popularity-matched pack with seed
`repeat-paging-20260830-v1`. Exclude the released public targets, all historical
consumed targets, and all three prior incremental-campaign splits, including
loose-title relatives. Raw targets, traces, and run artifacts remain ignored.
Only the 160-session screening split is opened initially.

Historical Cycle 5 and incremental-campaign packs reuse names such as
`cycle5_screening_0001` for different targets. Before passing consumed rows to
the unchanged target generator, `experiments/normalize_consumed_ids.py` prefixes
only `sample_id` with the source filename and records source and normalized
hashes. Ground truth and every target-selection field remain byte-equivalent at
the JSON-value level; this resolves provenance-name collisions without dropping
any exclusion.

## Decision rule

The primary correctness outcome is elimination of exact adjacent duplicate
slates on eligible stable-head turns beginning at turn 2. Report stable-head
opportunities, duplicates, highest-ranked-unseen selections, unique product
coverage, paired first-hit timing, target rank, official metrics, per-scenario
TechnicalScore, latency, tokens, and fallbacks.

Promote the candidate to one fresh confirmation comparison only if it:

- eliminates eligible exact duplicates without breaking override or changed-head
  resets;
- loses no more than `0.005` overall TechnicalScore or `0.010` HitRate;
- loses no scenario TechnicalScore by more than `0.020`;
- introduces no correctness/fallback failure and keeps warm p95 below one
  second and no more than twice control.

Because this is a user-facing correctness policy, a neutral or small score loss
inside those guards can be accepted with explicit justification. It must not be
described as a ranking improvement unless the score evidence supports that
claim. Confirmation must be non-negative to replace the locked finalist.
