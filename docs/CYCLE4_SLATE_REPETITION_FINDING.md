> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Cycle 4 finding: repeated slates waste most of every failed session

Recorded 28 August 2026 from the released-public baseline run `win-selected-repro`
(selected configuration, `source_changed_during_run: false`, TechnicalScore
`0.786724`, exactly reproducing the recorded macOS reference). This is a
development finding on already-consumed data. It is not a gate result and does
not promote anything.

## Observed behaviour

The agent ranks `candidate_limit` = 120 products internally and shows the first
`slate_size` = 10. The remaining 110 are computed, scored, reranked where
admitted, and then discarded every turn.

By roughly turn 3 the simulated customer exhausts its disclosable constraints and
begins answering "I don't have an additional preference". The query therefore
stops changing, the cache key `(query, revision)` stops changing, and the ranking
stops changing. The agent then re-serves a byte-identical slate until the turn
limit.

Measured over the 200 released-public sessions:

| Observation | Value |
|---|---:|
| Turns that repeated the previous slate verbatim | 176 |
| Repeated turns per failed session, mean | 7.0 |
| Distinct products ever shown per failed session, mean | 23.2 of 100 possible |
| Failed sessions whose target was never retrieved | 0 |
| Failed sessions whose target entered the 120-candidate pool | 21 of 21 |

Target internal ranks in the 21 failed sessions:

```
11, 11, 13, 14, 32, 33, 37, 40, 46, 48, 50, 51, 52, 52, 55, 65, 76, 81, 87, 93, 117
<=20: 4      <=40: 8      <=60: 15      <=80: 17
```

Example, `public_0076` turn 3: the shown slate holds 10 identifiers, `ranked_ids`
holds 120, the shown slate equals `ranked_ids[:10]` exactly, and the target sits
at internal position 11 — one slot past the cut, never displayed on any of the
remaining seven turns.

## Why repeating an unchanged slate cannot score

The official loop stops the session the moment the target appears in the scored
slate. If the session is still running, the target is not in the slate just
shown. When the ranking is also unchanged, the next slate is identical, so its
hit probability is exactly zero, not merely low. Those turns cannot contribute to
HitRate@10, MRR, or MTTC under any circumstances.

The customer's replies depend only on `ask_attribute`; `customer_reply` never
reads `recommendations`. Changing which products are displayed therefore cannot
change the dialogue, the extracted preferences, the query, or the ranking.

## The intent-override exception

A naive "always page forward when the ranking repeats" rule loses four sessions,
and all four are `intent_override`. The mechanism: a hit does not count until the
override has been delivered, so a target sitting in the top ten before the
override is shown but not yet scorable. Paging away from it before the gate opens
discards a session that the current behaviour would have won.

The organizer states that the preference change lands on turn 3 or 4, which the
official simulator implements as `rng.choice([3, 4])`. From turn 5 onward no gate
can still be closed, so paging is unconditionally safe there.

## Proposed change

Advance one page only when both hold:

1. this turn's ranking is identical to the previous turn's, and
2. the turn number is at least 5.

Any change in the ranking resets to page 0, so fresh information always restores
the strongest candidates immediately.

## Counterfactual result

The recorded traces contain `ranked_ids` for every turn, and the dialogue is
independent of the slate, so the alternative policy can be replayed exactly
rather than estimated. The replay of current behaviour reproduces all four
official metrics to six decimal places, which validates the method.

| Policy | TechnicalScore | Hit@10 | MRR | MTTC | Hits lost |
|---|---:|---:|---:|---:|---:|
| Current, no paging | 0.786724 | 0.8950 | 0.6137 | 3.245 | 0 |
| Page when unchanged, from turn 1 | 0.824840 | 0.9550 | 0.6275 | 3.045 | 4 |
| Page when unchanged, from turn 4 | 0.830076 | 0.9600 | — | — | — |
| **Page when unchanged, from turn 5** | **0.839176** | **0.9700** | **0.6459** | **2.980** | **0** |
| Anchor top 3, page from turn 5 | 0.823484 | 0.9550 | 0.6236 | 3.055 | 0 |
| Anchor top 2, page from turn 5 | 0.830380 | 0.9650 | 0.6279 | 3.025 | 0 |

Anchoring the head of the slate is worse than paging cleanly, because anchored
slots are spent on products already shown and already known not to be the target.

Under the turn-5 rule, 16 failed sessions convert to hits, at turns
`4,4,4,4,6,6,6,7,7,8,8,8,8,9,9,10` and slate positions
`1,1,1,2,2,2,3,3,4,4,5,5,6,7,8,10`. No session is lost and none is made slower.

## Status and limitations

- This is a counterfactual on the released public 200, which is fully consumed.
  It is descriptive only. It cannot promote anything and does not forecast
  organizer-private performance.
- The turn-5 threshold is derived from the published override rule, not fitted.
  The variant comparison above nevertheless used consumed data, and that
  selection pressure is disclosed rather than hidden.
- Depth experiments D60 and D120 both looked better on this same public set and
  then failed fresh screening. A public-set gain of this size is a reason to
  register a screening arm, not a reason to believe the number.
- The change costs no additional retrieval, model inference, memory, or latency;
  positions 11-120 are already computed and discarded today.
- Implementation, a registered protocol, and a screening run against the recorded
  C0 control are still outstanding.
