> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Realistic shopping merge result

Recorded 31 August 2026 after selectively integrating the accepted mechanisms
from `exp/06-repeat-driven-paging` and `final/mercury-incremental` into `main`.
This is consumed public-development evidence, not a private-test forecast.

## Selected design

The merged policy retains `main`'s earliest repeat paging and explicit override
reset. It adds stable Top-10 membership detection and highest-ranked-unseen
selection from the incremental finalist. On an override, correction, relaxation,
polarity change, replacement, or category change, it clears exposure history and
replays the new first page. It does not enable the rejected intent-conditioned
ranking experiment.

The same merge also carries target-independent shopping semantics: local hard
and soft cue scope, soft exclusions, continuous soft-budget proximity, semantic
state deltas, and confidence-gated buying/browsing intent. These code paths were
present in both matched arms; only `repeat_driven_paging` differed in the public
comparison.

## Matched public comparison

Both arms used the 200-row released public development set, the same catalog,
models, source checkout and evaluator. Source did not change during the run and
neither arm produced a fallback or agent error.

| Metric | Guarded early paging | Merged realistic paging | Delta |
|---|---:|---:|---:|
| HitRate@10 | 0.970000 | **0.980000** | +0.010000 |
| MRR | **0.641633** | 0.640647 | -0.000986 |
| MTTC | 2.905000 | **2.860000** | -0.045000 |
| Efficiency | 0.809500 | **0.814000** | +0.004500 |
| TechnicalScore | 0.839390 | **0.844994** | +0.005604 |
| Prompt tokens | 2,372,103 | 2,372,103 | 0 |
| Evaluated turns | 590 | **568** | -22 |
| Suite p95 turn latency | **0.416 s** | 0.574 s | +0.158 s |

The candidate recovered two previous misses, one browsing and one buying, while
preserving all earlier hits. Eighteen session outcomes changed. The result found
196 of 200 targets; its remaining four failures were ranking/admission or policy
misses, not retrieval misses.

## Behaviour audit

Across the public traces, exact adjacent duplicate slates fell from 39 to 7.
Every remaining duplicate was intentional: it occurred immediately after a
detected intent override, when replaying page one protects products shown before
the new intent became eligible. The merged arm selected an unseen-product page
112 times, performed 57 semantic resets, and every reset returned page zero.

The MRR decrease remains a real trade-off. Some targets appeared sooner at a
lower rank, which is preferable for catalogue exploration but less favoured by
the evaluator's reciprocal-rank weight. The two recovered misses and lower MTTC
more than offset that change. The higher sequential p95 is disclosed; token and
model work were identical, and the selected run remained below the local
turn-latency ceiling.

Receipts are retained in:

- `runs/realistic-merge-public-20260831/`
- `runs/realistic-merge-public-audit-20260831/`
- `runs/realistic-merge-final-reproduction-20260831/`
- `runs/realistic-merge-private-like-20260831/`

The promotion is narrow: it validates the merged public behaviour on a consumed
development set. The parser and intent improvements passed all 19 assertions in
the authored private-like robustness pack, but their true private-set
generalization remains unobserved.
