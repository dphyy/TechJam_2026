# Mercury incremental improvement campaign

Registered 30 August 2026 before any campaign score-bearing run or runtime
change. The campaign evaluates parser correctness, override-aware slate novelty,
typed ranking, structured price handling, and confidence-gated intent diversity.

## Branch and selection policy

Every stage is implemented on its own `exp/NN-name` branch. A stage branches
from the latest accepted baseline. Rejected branches remain available for audit
but are not inherited by later stages. Each branch records tests, a source-matched
control, candidate results, paired session changes, resource measurements, and a
keep or reject decision.

Correctness changes may be accepted with a neutral or small score decline when
they fix a predeclared language-state defect, introduce no new capability failure,
and lose no more than `0.005` TechnicalScore or `0.010` HitRate on screening.
Score-oriented changes require at least `+0.010` TechnicalScore on screening,
no more than `-0.010` HitRate overall, no scenario TechnicalScore loss greater
than `0.020`, no correctness regression, warm p95 below one second, and no more
than twice the source-matched control runtime. Confirmation requires a
non-negative overall delta. Validation is opened once for the frozen combined
finalist and is consumed regardless of outcome.

Public-set runs are descriptive regression checks only and never select a
candidate.

## Campaign target lock

The committed Cycle 2, Cycle 3, and Cycle 5 generators reconstructed their
historical deterministic target packs. The new pack excludes the 200 released
public targets, all 704 reconstructed prior synthetic targets, and every loose
title-family relative detected by the generator.

- Generator: `experiments/cycle5_prepare.py`
- Seed: `mercury-campaign-20260830-v1`
- Screening: 160 sessions (64 buying, 64 browsing, 24 intent override, 8 boundary)
- Confirmation: 80 sessions (32 buying, 32 browsing, 12 intent override, 4 boundary)
- Validation: 80 sessions with the same 40/40/15/5 mix
- Screening SHA-256: `0e02eded26729f33527a55b3c9ec00e4720db535b2c7e40e1a9e88505ade2448`
- Confirmation SHA-256: `f9a806dc13739a5cf4f3e343c6bf76ac5d8ae2cb4ec9564a931510377c617ce8`
- Validation SHA-256: `d1faf26588566007e558eb858681463c9ca747e890931bcf3e52d99a864c2bd8`

The catalog no longer has an unused sample of the organizer's extreme
popularity tail after excluding prior target families. The new pack therefore
contains no `20000+` targets and shifts the unmet upper-tail quota primarily
into `1000-5000`. Results are synthetic same-simulator engineering evidence,
not organizer-private or real-user evidence.

The raw target rows, manifests, traces, model assets, and run outputs stay local
and ignored by Git.

## Execution order

1. Freeze and reproduce the selected campaign control.
2. Remove parser cue leakage from open-vocabulary residuals.
3. Scope hard and soft cues to the value they modify.
4. Separate hard and soft negative preferences.
5. Expose explicit additive/refinement/replacement/polarity/category state deltas.
6. Compare base Top-10 membership rather than the complete 120-item order.
7. Select highest-ranked unseen products during stable-head paging.
8. Test override-aware paging beginning before turn 5.
9. Compute typed evidence ranking in shadow mode, then activate only supported
   hard/soft consumers that pass the gate.
10. Test continuous structured price proximity without putting budget numbers
    into lexical retrieval.
11. Measure an evaluator-only oracle intent-diversity ceiling, then test the
    actual confidence-gated intent policy only if the ceiling is positive.
12. Combine accepted stages, freeze the finalist, run confirmation, and open
    validation once only if confirmation passes.

No runtime feature may read sample IDs, scenario labels, ground truth, future
turns, or evaluator-owned diagnostics. Oracle intent analysis is evaluator-only
and can never become a submission configuration.

## C0 screening baseline

Run `c0-screening` used the exact `campaign_control.json` bytes on the locked
160-session screening split. Source did not change during the run and there were
zero startup or turn fallbacks.

| HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---:|---:|---:|---:|---:|---:|
| 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.712695 s |

Ever-observed ranked recall was `0.86875` at 10, `0.88750` at 30,
`0.93750` at 60, and `0.96250` at 120. The ten official misses comprise two
targets never retrieved and eight ranking-or-policy misses; seven missed targets
never entered the reranker prefix. These diagnostics are evaluator-owned and
are not available to runtime behavior.
