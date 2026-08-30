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

## Stage 1A: parser cue cleanup — keep

Branch `exp/01a-parser-cue-cleanup`, implementation commit `25ed673`, removes
preference discourse cues such as `preferably` and `leaning toward` from the
open-vocabulary lexical residual. A focused regression test proves that useful
values (`waterproof`, `canvas`) remain in the query while the cue words do not.
The complete suite passed with 525 tests and Ruff passed.

The source-matched screening candidate was exactly neutral:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| C0 control | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.712695 s |
| Stage 1A | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.557279 s |

All 491 paired turn responses were identical. The lexical query, preferences,
ranked IDs, and slate page also had zero paired changes, showing that the frozen
simulator did not exercise these cue forms. The lower observed latency is treated
as run-to-run noise, not an improvement. This stage is kept as a targeted,
tested correctness fix with no screening or resource regression.

## Stage 1B: local hard/soft scope — keep

Branch `exp/01b-local-hard-soft-scope`, implementation commit `f83de65`, assigns
each explicitly extracted value the nearest hard or soft cue in its clause. This
corrects mixed-force input such as `I need boots that would ideally be blue`:
`boots` remains hard while `blue` becomes a soft preference. It also recognizes
`preferably`, `mandatory`, and `non-negotiable` without leaking those cues into
the lexical query. Negative requirements are intentionally unchanged until
Stage 1C. The complete suite passed with 527 tests and Ruff passed.

The source-matched screening candidate was again exactly neutral:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1A control | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.557279 s |
| Stage 1B | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.593333 s |

Across all 491 paired turns there were zero changes to responses, queries,
preferences, preference groups, intent, retrieval plans, ranked IDs, or slate
pages. This exposes a frozen-simulator coverage gap: the generated language does
not test mixed strength within one clause. The stage is kept as a predeclared,
unit-tested correctness fix with no score, fallback, or resource regression;
latency differences are treated as noise.

## Stage 1C: hard versus soft negatives — keep

Branch `exp/01c-soft-negative-preferences`, implementation commit `fa91825`,
keeps direct exclusions such as `no leather` as hard constraints while parsing
soft language such as `prefer not to have leather` as a soft negative. Both stay
out of lexical retrieval. Hard negatives retain the pool-wide constraint guard;
soft negatives are represented as `Prefer to avoid` and receive a bounded
`0.02` evidence demotion without filtering or changing candidate identity.
Repeated application is idempotent. The complete suite passed with 532 tests
and Ruff passed.

The source-matched screening candidate was exactly neutral:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1B control | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.593333 s |
| Stage 1C | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.581618 s |

All 491 paired responses, preferences, hard/soft groups, plans, rankings, and
slate pages were identical, and the soft-negative adjustment fired on zero
turns. The frozen simulator has no actual soft-exclusion paraphrases; incidental
catalog phrases containing `No` are not user exclusions. The stage is kept as a
predeclared correctness and preference-fidelity fix with no regression. The
`0.02` weight is bounded and unit-tested but cannot be claimed as empirically
tuned until a dedicated soft-negative language pack exists.

## Stage 1D: semantic state deltas — keep

Branch `exp/01d-semantic-state-deltas`, implementation commit `b153336`, exposes
an immutable delta after every message. It records added and removed
`(attribute, value, polarity)` facts, whether replacement syntax was explicit,
and one target-independent update kind: `none`, `refinement`, `additive`,
`replacement`, `polarity_change`, or `category_change`. It is diagnostic-only in
this stage and cannot inspect evaluator labels or future turns. The complete
suite passed with 534 tests and Ruff passed.

The source-matched screening candidate was exactly neutral:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1C control | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.581618 s |
| Stage 1D | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.538604 s |

All 491 paired responses, state, plans, rankings, and slate pages were identical.
Observed updates were 317 refinements, 131 no-ops, 23 replacements, 10 category
changes, 5 additive changes, and 5 polarity changes. Of the 24 generated
`ignore my earlier preference` override messages, 20 were replacements, one was
a category change, and three were correctly no-ops because the requested value
did not alter the active ledger. This stage is kept as tested infrastructure for
override-aware paging; the observed latency difference is treated as noise.

## Stage 2A: Top-10 membership stability — keep

Branch `exp/02a-top10-set-stability`, implementation commit `7c1b435`, defines a
stable slate as equal base Top-10 membership rather than equality of the full
120-item ranking. Rank swaps inside the head and tail-only changes now advance;
changing even one head member resets to page zero so newly relevant items are
shown. Page slicing and the turn-5 threshold remain unchanged until later
stages. Focused tests cover head reordering, tail reordering, and one-member head
replacement. The complete suite passed with 535 tests and Ruff passed.

The source-matched screening candidate was exactly neutral:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1D control | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.538604 s |
| Stage 2A | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.533357 s |

All 491 paired responses and slate pages were identical. In this pack, every
non-cached full-ranking change that reached paging also changed Top-10
membership; cached rankings were fully equal. The stage is kept as the requested
unit-tested paging correctness fix with no score, fallback, or resource
regression. It is not credited with a measured campaign improvement.

## Stage 2B: highest-ranked unseen paging — keep

Branch `exp/02b-highest-ranked-unseen`, implementation commit `261c92b`, tracks
products actually shown during the session. Page zero still serves the current
Top 10, including legitimate overlap after a changed-head reset. An advanced
page now serves the highest-ranked currently unseen products instead of a fixed
offset; when none remain it holds the last non-empty page. The complete suite
passed with 537 tests and Ruff passed.

The screening candidate improved discovery while preserving resources:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2A control | 0.937500 | 0.575444 | 3.131250 | 0.798758 | 1,826,142 | 0.533357 s |
| Stage 2B | 0.943750 | 0.574660 | 3.100000 | 0.802273 | 1,826,142 | 0.557221 s |

All 94 advanced turns selected entirely unseen products and no exhaustion hold
was needed. Fifty-five paired slates changed, producing five target gains and
zero paired target losses; one prior miss became a turn-10 boundary hit. The
buying and boundary scenario TechnicalScores improved, intent-override was
unchanged, and browsing declined only `0.001753`, inside the predeclared
`0.020` scenario guard. Overall TechnicalScore gained `0.003515`, HitRate gained
`0.006250`, fallbacks remained zero, and runtime stayed comparable. Although the
gain is below the score-feature promotion threshold, this stage is kept under
the correctness rule because it guarantees novelty and improves aggregate
discovery without a material regression.

## Stage 2C trial: override-aware paging from turn 2 — reject

Branch `exp/02c-override-aware-early-paging`, commits `7442bbf` and `9921748`,
tested the most aggressive proposal: advance on stable Top-10 membership from
turn 2, but force page zero for replacements, polarity changes, category
changes, and explicit replacement language. Explicit replacement is preserved
even when the parser cannot form an assertion or the ledger does not change.
The complete suite passed with 539 tests and Ruff passed.

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2B control | 0.943750 | 0.574660 | 3.100000 | 0.802273 | 1,826,142 | 0.557221 s |
| Stage 2C turn 2 | 0.943750 | 0.569660 | 3.018750 | 0.802398 | 1,821,316 | 0.704068 s |

All 24 explicit simulator override messages reset to page zero. Early novelty
improved MTTC by `0.081250` turns, but did not add a hit, reduced MRR by
`0.005000`, and improved overall TechnicalScore by only `0.000125`. Boundary
TechnicalScore declined `0.027500`, exceeding the predeclared `0.020` scenario
guard. Turn-2 paging was therefore rejected as too aggressive, motivating the
turn-3 trial below.

## Stage 2C: override-aware paging from turn 3 — keep

Branch `exp/02c2-override-aware-turn3`, implementation commits `fdddd2b`,
`6997db4`, and candidate commit `65934d7`, advances a stable Top-10 from turn 3
while forcing page zero for replacements, polarity changes, category changes,
and explicit replacement language. Message-level detection preserves an
override even when the parser forms no assertion. The complete suite passed
with 540 tests and Ruff passed.

The source-matched screening result was:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2B control | 0.943750 | 0.574660 | 3.100000 | 0.802273 | 1,826,142 | 0.557221 s |
| Stage 2C turn 3 | 0.943750 | 0.574660 | 3.025000 | 0.803773 | 1,826,142 | 0.581242 s |

All 24 explicit overrides reset to page zero. Twelve sessions found the same
target one turn earlier with identical hit status and best rank. MTTC improved
by `0.075000`, TechnicalScore improved by `0.001500`, and MRR and HitRate were
unchanged. Scenario TechnicalScore deltas were non-negative: boundary
`0.000000`, browsing `+0.001875`, buying `+0.001562`, and intent override
`+0.000833`. The sibling turn-2 branch was rejected because its boundary score
fell `0.027500`. Turn 3 is kept under the correctness rule: it implements stable
slate novelty and explicit override reset with no measured quality regression,
although its score gain is not large enough to claim a ranking improvement.

## Stage 3A: typed-plan shadow scoring — keep as experimental infrastructure

Branch `exp/03a-typed-plan-shadow`, implementation commit `5e856c0`, consumes
the typed hard/soft plan as bounded catalog evidence while preserving candidate
scores and order in `shadow` mode. Budget is excluded for the dedicated Stage 4
test. Diagnostics expose base candidate scores, typed evidence, and applied
adjustments; the latter are all zero in shadow mode. The complete suite passed
with 543 tests and Ruff passed.

The source-matched shadow run was output-identical:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2C control | 0.943750 | 0.574660 | 3.025000 | 0.803773 | 1,826,142 | 0.581242 s |
| Stage 3A shadow | 0.943750 | 0.574660 | 3.025000 | 0.803773 | 1,826,142 | 0.714010 s |

All 475 responses, ranked ID lists, and slate pages were identical. Typed
evidence was nonzero on every turn. An evaluator-only sweep over 315 turns where
the target was ranked was unfavorable: at the predeclared `0.10` weight, target
rank improved 38 times, worsened 30 times, stayed equal 247 times, produced zero
Top-10 gains and one Top-10 loss, and reduced mean target reciprocal rank by
`0.002251`. Weights from `0.02` through `0.20` all had negative mean target-RR
deltas. Shadow mode is kept as reusable experimental infrastructure, not enabled
in the finalist. The predeclared active weight still receives one end-to-end run
on a separate branch because static rank simulation cannot reproduce dialogue
and paging feedback.

## Stage 3B: active typed-plan scoring — reject

Branch `exp/03b-typed-plan-active`, candidate commit `1e75314`, activated the
predeclared `0.10` typed-plan adjustment after neural ranking and reapplied the
hard-constraint guard. The complete suite passed with 544 tests and Ruff passed.

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 3A shadow/control | 0.943750 | 0.574660 | 3.025000 | 0.803773 | 1,826,142 | 0.714010 s |
| Stage 3B active | 0.943750 | 0.578065 | 3.031250 | 0.804670 | 1,832,671 | 0.673366 s |

No hit was gained or lost. MRR improved `0.003405`, but MTTC worsened
`0.006250`, token use rose by 6,529 because one session took an extra turn, and
overall TechnicalScore gained only `0.000897`. Boundary TechnicalScore declined
`0.003125`; other scenario deltas were positive but all below `0.0021`. The
candidate was rejected because it fell far short of the predeclared `+0.010`
score-feature gate and the shadow sweep showed no robust weight region. The
typed scorer remains available behind `typed_plan_mode`; the finalist keeps it
explicitly `off`.

## Stage 4: structured budget proximity — keep

Branch `exp/04-structured-budget-proximity`, implementation commit `b2f138a`,
keeps firm ceilings, floors, and ranges as binary fit evidence while ranking
explicitly soft budget figures by continuous price proximity. Missing prices and
inconclusive `from` lower bounds remain neutral, adjustments are bounded by the
existing `0.02` weight, candidate identity is preserved, and repeated
application is idempotent. Budget numbers remain outside lexical/BM25 retrieval.
The complete suite passed with 546 tests and Ruff passed.

The source-matched screening result was exactly neutral:

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2C control | 0.943750 | 0.574660 | 3.025000 | 0.803773 | 1,826,142 | 0.581242 s |
| Stage 4 | 0.943750 | 0.574660 | 3.025000 | 0.803773 | 1,826,142 | 0.665875 s |

All 475 responses and rankings were identical. The pack contained zero parsed
budget turns and therefore zero price adjustments. This stage is kept as a
focused, unit-tested correctness improvement with no screening regression, not
as evidence of score benefit. A future budget-specific pack must measure target
proximity, hard-limit safety, uncertain-price neutrality, and sensitivity to the
`0.02` weight before it is tuned further.

## Stage 5A: evaluator-only intent-diversity oracle — keep as infrastructure

Branch `exp/05a-intent-diversity-oracle`, implementation commit `1aa047e`, adds
a deterministic facet-diversity reranker and an evaluator-only script. The
reranker anchors the leader and greedily balances rank prior with supported
category, material, color, and style novelty inside a 30-item prefix. Missing
facets earn no novelty. The oracle alone reads `scenario_type`; runtime code
cannot access labels, targets, or future turns. The complete suite passed with
549 tests and Ruff passed.

On 112 target-ranked browsing observations, the oracle ceiling was positive but
small. Strength `0.20` improved target rank 15 times, worsened it 14 times, was
unchanged 83 times, created two Top-10 gains and one Top-10 loss, and improved
mean target reciprocal rank by `0.001472`. Strength `0.40` had a larger mean-RR
gain but worsened more observations than it improved; `0.50` had equal Top-10
gains and losses. Strength `0.20` is therefore the only runtime candidate. The
oracle is kept as experimental infrastructure and can never be a submission
configuration. A separate active branch must use only the live classifier and a
predeclared confidence gate.

## Stage 5B: confidence-gated intent diversity — reject from finalist

Branch `exp/05b-confidence-gated-intent-diversity`, implementation commit
`02c43cb`, enabled the Stage 5A reranker only when the live classifier reported
`browsing` with confidence at least `0.65`. It used the predeclared `0.20`
strength and 30-item pool, anchored the leader, and stopped before the first
constraint- or object-penalized candidate. The complete suite passed with 552
tests and Ruff passed.

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 4 control | 0.943750 | 0.574660 | 3.025000 | 0.803773 | 1,826,142 | 0.665875 s |
| Stage 5B | 0.943750 | 0.577282 | 3.031250 | 0.804435 | 1,831,950 | 0.759490 s |

TechnicalScore improved only `0.000662`, far below the score-feature gate;
HitRate was unchanged and MTTC worsened `0.006250`. Seven sessions changed:
four reciprocal ranks improved, three worsened, no hit was gained or lost, and
one target arrived a turn later. Browsing was essentially neutral
(`+0.000008` scenario TechnicalScore); most aggregate gain came from buying
(`+0.001563`), where diversity should not have been active.

The gate applied in 27 sessions, but only 14 were labelled browsing; it also
fired in six buying, five intent-override, and two boundary sessions. It reached
only 14 of 64 browsing sessions. Labels here are synthetic evaluator metadata,
so this is diagnostic rather than a production accuracy estimate, but the gain
clearly depends on off-intent activation. Runtime diversity is rejected. A
future retry should first calibrate intent routing on a dedicated,
human-reviewed set.
