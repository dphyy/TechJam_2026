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

## Screening result — behavior succeeds, promotion gate fails

The fresh screening lock contains 160 sessions and has SHA-256
`2dfae2b1cd153cdf97d219a2b0c47cb88a827b9d430c11b45b6024692cc697b6`.
After all prior target-family exclusions, the two highest popularity bands had
no eligible families; 238 of 320 total locked targets were therefore drawn from
the `1000-5000` band. This known population limitation applies equally to both
arms.

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Turn-3 control | 0.937500 | 0.594750 | 2.937500 | 0.808425 | 1,891,213 | 0.703819 s |
| Turn-2 repeat-driven | 0.937500 | 0.588500 | 2.925000 | 0.806800 | 1,877,488 | 0.746326 s |

The candidate eliminated all eight exact adjacent duplicate slates observed on
eligible stable-head turns (`8 → 0`) and converted them to highest-ranked-unseen
selections (`85 → 93`). Mean unique products exposed through turn 2 increased
from `16.231250` to `16.731250`; through turn 3 it increased from `18.537500`
to `18.956250`. Override resets remained exactly 52, HitRate was unchanged,
tokens fell by 13,725, and both arms had zero fallbacks.

Only two of 160 session outcomes changed. Both were boundary sessions in which
the target was found at rank 2 on turn 2 instead of rank 1 on turn 3. Thus the
candidate genuinely found both targets one turn earlier, but MRR fell by
`0.006250`. Official TechnicalScore fell `0.001625`; its paired bootstrap 95%
interval was `[-0.004063, 0.000000]`. Buying, browsing, and intent-override
scenario scores were identical. Boundary TechnicalScore fell `0.032500` over
only eight cases, exceeding the registered `0.020` scenario guard.

The product-logic hypothesis is supported: repeating an unchanged slate was
wasteful, and immediate unseen paging removed that waste without losing a hit.
The evaluator nevertheless prefers the later rank-1 placement because the MRR
gain is weighted more heavily than finding the item one turn earlier. This same
boundary-score tension appeared in the earlier consumed turn-2 trial, so it is
not dismissed as a single-run anomaly.

Under the predeclared protocol the candidate does not open confirmation and
does not replace `configs/selected.json`. It remains available as the explicit
`configs/exp06_repeat_driven_paging.json` product-policy alternative. Promoting
it would be a conscious product-utility override—prioritizing no duplicate
slates and earlier discovery over the competition metric—not an evidence-backed
TechnicalScore improvement.

## Subsequent product-policy decision

On 31 August 2026, the team explicitly chose the product-utility override
described above. `configs/selected.json` now sets
`slate_paging_first_turn = 2`, eliminating an unchanged Top-10 repeat at the
earliest possible opportunity. The historical screening result and failed
competition-score promotion gate remain unchanged: this is a deliberate product
policy choice, not a claim that the candidate improved TechnicalScore.
