# Progressive frontier reranking protocol

Registered on 30 August 2026 before the new target pack was generated and
before runtime behavior changed.

## Hypothesis

The selected runtime repeatedly reuses one neural Top-30 after the preference
state stops changing. A failed session can therefore page through candidates
that were retrieved but never compared by the neural model. The proposed
mechanism keeps the existing 30-pair per-turn ceiling and progressively scores
one new, unseen batch only after an uninformative or rejected turn. A companion
seen-aware slate selects the highest-ranked products not previously displayed.
An intent override clears displayed-product history because earlier products
may only become eligible after the correction.

The mechanism may use active state, current candidates, prior model scores,
shown identifiers, and ordinary catalog evidence. It may not read sample IDs,
targets, scenario labels, future turns, or evaluator outcomes at runtime.

## Locked data

Use the existing popularity-matched Cycle 5 preparation method with seed
`frontier-20260830-v1`. Exclude the released public targets and every previously
consumed synthetic target dataset available in the repository. The generated
screening, confirmation, and validation targets must be mutually disjoint and
must have zero exact or loose-title-family overlap with excluded data.

Only screening outcomes may select between these fixed arms:

1. `control`: current selected behavior.
2. `seen-aware`: current ranking, but never repeat a displayed product unless
   the bounded pool is exhausted; clear display history on override.
3. `frontier`: seen-aware selection plus one new neural batch on an unchanged,
   uninformative or rejected turn. Each candidate is scored at most once per
   state revision.

No thresholds, batch sizes, phrase rules, or ranking weights may be changed
after viewing screening outcomes. At most one candidate can advance.

## Gates

Screening promotion requires all of the following against the source-matched
control:

- Technical Score improvement of at least `0.005`.
- No HitRate@10 loss and no MRR loss greater than `0.005`.
- Zero agent errors, fallbacks, invalid IDs, source drift, or correctness-test
  regressions.
- Warm p95 turn latency no more than `1.25x` control and total neural tokens no
  more than `2x` control.

If an arm passes, freeze source and configuration before one confirmation run.
Final promotion requires non-negative confirmation Technical Score and HitRate,
no MRR loss greater than `0.005`, and the same correctness/resource gates. The
validation split remains unopened unless a later final-release protocol uses it.

The consumed public set may be run once after selection only as a descriptive
non-regression check. It cannot rescue a candidate that fails fresh screening
or confirmation.

## Locked pack

The pack was generated with the registered seed and verified before runtime
behavior was edited. Its split sizes and SHA-256 hashes are:

| Split | Sessions | SHA-256 |
|---|---:|---|
| Screening | 160 | `cc9dde490c0aef4f270d2e3cebeaa8222b83a355dae51fa3320848e348262c38` |
| Confirmation | 80 | `02fe8eb924db2fee360889483acd9f8f000c8c08d6b992355efbad413459d01a` |
| Validation | 80 | `4565cd146fdce6b664589140e18374426c1589422e2279f95c4ab88729b0fa69` |

## Screening result

All three arms ran from the same source tree and completed with zero fallback
turns and zero agent errors.

| Arm | Technical Score | HitRate@10 | MRR | p95 seconds | Prompt tokens | Decision |
|---|---:|---:|---:|---:|---:|---|
| Control | 0.797735 | 0.918750 | 0.618284 | 0.351341 | 1,953,123 | Baseline |
| Seen-aware | 0.795676 | 0.906250 | 0.633504 | 0.411396 | 1,960,771 | Reject |
| Frontier | 0.802093 | 0.912500 | 0.639477 | 0.463785 | 2,348,043 | Reject |

The frontier arm improved Technical Score by `0.004358`, MRR by `0.021193`,
and ever-reranked-prefix recall from `0.91875` to `0.95625`. It did not satisfy
the registered minimum score gain, no-HitRate-loss requirement, or p95 latency
ceiling. Seen-aware selection by itself also lost HitRate. Neither candidate
advanced to confirmation, and the confirmation and validation splits remain
unopened. Both mechanisms remain opt-in diagnostic configurations; the selected
runtime is unchanged.
