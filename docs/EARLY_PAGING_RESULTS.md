> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Early-paging comparison result

Decision: **Rejected without an override reset.**

This record covers the original unguarded candidate. A separately registered
follow-up added an intent-override page reset and was later promoted; see
`docs/EARLY_PAGING_OVERRIDE_RESET_RESULTS.md`.

The comparison followed `docs/EARLY_PAGING_PROTOCOL.md`. Both arms used the
same source snapshot, catalog and 200-session public development set. Neither
source tree changed during its run.

| Metric | Turn-5 control | Turn-1 candidate | Candidate delta |
|---|---:|---:|---:|
| HitRate@10 | 0.970000 | 0.960000 | -0.010000 |
| MRR | 0.645919 | 0.632466 | -0.013453 |
| MTTC | 2.980000 | 3.010000 | +0.030000 |
| Efficiency | 0.802000 | 0.799000 | -0.003000 |
| TechnicalScore | 0.839176 | 0.829540 | -0.009636 |
| p95 turn latency | 0.443610 s | 0.439833 s | -0.85% |
| Prompt tokens | 2,375,969 | 2,378,735 | +2,766 |
| Fallback / agent-error turns | 0 / 0 | 0 / 0 | 0 / 0 |

Buying and browsing HitRate and MRR were unchanged, with slightly earlier
turns. The regression was concentrated in intent overrides:

| Intent-override metric | Turn-5 control | Turn-1 candidate | Candidate delta |
|---|---:|---:|---:|
| HitRate@10 | 0.900000 | 0.833333 | -0.066667 |
| MRR | 0.748611 | 0.687500 | -0.061111 |
| MTTC | 4.533333 | 5.233333 | +0.700000 |

## Why the ranking-reset assumption failed

An override does not always change the ranking. In several sessions the
simulator's explicit override restated a material that had already been revealed
on an earlier turn. The target was visible in ranks 1–10 before the override but
could not legally score yet. Because the final ranking was unchanged at the
override turn, turn-1 paging moved to ranks 11–20 and hid the now-eligible target.

The candidate lost control wins in `public_0071`, `public_0103` and
`public_0183`, whose targets were at ranks 1, 2 and 1 on the override turn. It
gained the previous miss `public_0144` by reaching a deeper page, for a net loss
of two sessions. This directly validates the original eligibility concern.

Complete receipts are retained under:

- `runs/early-paging-control-20260830/`
- `runs/early-paging-candidate-20260830/`

`configs/paging_from_start.json` remains as a reproducible rejected candidate.
The unguarded configuration is not used by the public entrypoint.
