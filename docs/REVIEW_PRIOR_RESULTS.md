# Corrected review prior: results and selection

Recorded 31 August 2026. Decision: promote the mixed two-stage prior in `configs/selected.json`; retain both constraint checks.

## What is selected

The signal is half bounded log review-count popularity and half confidence-adjusted star quality. Review count saturates at 500,000; quality is `(stars - 3) / 2 * count / (count + 20)`. Missing/invalid metadata is neutral. These are weak priors, not evidence that popularity guarantees suitability.

The admission adjustment is capped at 0.30, applied after existing constraints/preferences and before truncation to 120. It can also improve admission to the unchanged neural top-30. A separately capped 0.02 adjustment operates after neural score replacement and before the final constraint check. The old numerical bonus is removed when neural scores replace the old scale; repeat calls on the same scale replace rather than accumulate it. Without neural score replacement, the second application is skipped. Existing guarded score separation is protected.

No target IDs, evaluation labels, or public-message templates enter runtime logic. No review text is appended to model prompts. Neural model, blend 0.75, top-30 budget, questions, and paging stay fixed.

## Signal selection: Cycle 5 screening, admission-only

| Signal | Hit@10 | MRR | MTTC | TechnicalScore | Tokens |
|---|---:|---:|---:|---:|---:|
| control | 0.956250 | 0.606372 | 3.268750 | 0.814662 | 1,942,690 |
| count | 0.968750 | 0.619814 | 3.012500 | 0.830069 | 1,922,454 |
| raw_stars | 0.950000 | 0.594263 | 3.206250 | 0.809154 | 1,924,237 |
| stars | 0.968750 | 0.605439 | 2.993750 | 0.826132 | 1,918,489 |
| mixed | 0.968750 | 0.626565 | 3.006250 | 0.832220 | 1,916,148 |

Raw average stars lost score and hit rate. Confidence adjustment helped; mixed was the best observed screening variant. Its 0.002150 paired lead over count-only is small: 10,000-resample paired bootstrap 95% interval [-0.004714, 0.009239]. This is not proof of universal superiority.

## Does a second prior application help?

| Placement of mixed prior | Pre weight | Post weight | Screening TechnicalScore |
|---|---:|---:|---:|
| Admission only | 0.30 | 0 | 0.832220 |
| Final only | 0 | 0.02 | 0.826065 |
| Both (selected) | 0.30 | 0.02 | 0.837746 |

Both-stage use adds 0.005526 over admission-only at equal Hit@10. The two stages have distinct admission and final-ordering roles; this is deliberate repeated use of evidence, not accidental numerical accumulation.

## Frozen candidate across datasets

| Dataset | n | Control score | Selected score | Delta | Control Hit@10 | Selected Hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| Public | 200 | 0.844994 | 0.866792 | +0.021798 | 0.980000 | 0.990000 |
| Cycle 5 screening | 160 | 0.814662 | 0.837746 | +0.023084 | 0.956250 | 0.968750 |
| Cycle 5 confirmation | 80 | 0.841700 | 0.869153 | +0.027453 | 0.950000 | 0.987500 |
| Cycle 5 validation | 80 | 0.834314 | 0.839479 | +0.005165 | 0.975000 | 0.987500 |
| Cycle 3 screening | 160 | 0.842194 | 0.843946 | +0.001752 | 0.968750 | 0.968750 |

| Dataset | Control MRR | Selected MRR | Control MTTC | Selected MTTC | Control tokens | Selected tokens |
|---|---:|---:|---:|---:|---:|---:|
| Public | 0.640647 | 0.683974 | 2.860000 | 2.670000 | 2,372,103 | 2,370,993 |
| Cycle 5 screening | 0.606372 | 0.644152 | 3.268750 | 2.993750 | 1,942,690 | 1,908,468 |
| Cycle 5 confirmation | 0.685665 | 0.694678 | 2.950000 | 2.650000 | 987,658 | 960,452 |
| Cycle 5 validation | 0.614380 | 0.592431 | 2.875000 | 2.600000 | 944,560 | 893,594 |
| Cycle 3 screening | 0.647314 | 0.656488 | 2.818750 | 2.868750 | 1,719,765 | 1,768,180 |

The public run recovers two misses and loses no earlier hit. Confirmation recovers three and validation one, also without losing an earlier hit. Cycle 3 hit outcomes are unchanged. Validation MRR decreases despite better hit rate and turn efficiency. On the lower-popularity Cycle 3 counter-distribution, the score gain is small and tokens increase by 2.8%; do not claim universal efficiency gains.

All runs have zero startup fallbacks, fallback turns, agent errors, and source drift during evaluation. The control reproduces main's 0.844994 public score. Public selected p95 is 0.356 seconds; latency variation is not a controlled hardware benchmark.

Paired 10,000-resample bootstrap intervals for candidate-minus-control TechnicalScore:
- `review-prior-cycle3-screening-20260831`: [-0.007981, 0.011802].
- `review-prior-cycle5-confirmation-20260831`: [-0.003845, 0.062737].
- `review-prior-cycle5-validation-20260831`: [-0.022881, 0.037941].
- `review-prior-public-20260831`: [0.008215, 0.036998].

The small confirmation, validation, and Cycle 3 intervals cross zero. Report the observed gains, not certainty about unseen organizer-private performance.

## Constraint checks: score versus shopping correctness

Both checks use deterministic local evidence, not model calls. The first protects the candidate pool before truncation; the second repairs constraint ordering after neural fusion replaces scores and discards the old constraint penalty.

| Checks enabled, same mixed both-stage prior | Screening score | Tokens | Returned observed contradictions |
|---|---:|---:|---:|
| Both | 0.837746 | 1,908,468 | 0 |
| Pre only | 0.837746 | 1,908,468 | 0 |
| Post only | 0.840043 | 1,901,686 | 3 |
| Neither | 0.841141 | 1,901,686 | 52 |

On public, both and pre-only also tie exactly at 0.866792 and 2,370,993 tokens. Thus the second check has **no measured public or screening score/token benefit** in this comparison. Its mean local cost on public is about 0.75 ms per retrieval, and a regression test demonstrates a neural ranker reintroducing an excluded material unless it is checked again. It remains a cheap correctness guard. The selected screening post-check reordered six candidate lists, below the outcome-sensitive positions.

Removing the pre-check slightly improves this synthetic score but worsens constraint compliance. It is not a production improvement. These checks demote contradictions rather than hard-filtering them: even with both checks, confirmation returns 12 observed contradictions, down from 22 in the control. This residual limitation is not hidden behind the zero-error result.

The confirmation trace audit reproduces the score and localizes all 12 marked
contradictions to two first-page slates. One pool has no uncontradicted products;
the other has only eight, and the ten-item slate fills the rest. All 12 still
carry their constraint penalties: popularity did not undo them. Existing broad
negation scope incorrectly interprets phrases such as "no-tie lacing" and "style
without compromise" as exclusions of surrounding product terms. These are
contradictions against the parsed state, not independently verified human intent.
Negation-scope repair and short-slate/clarification behavior are separate follow-up
work; they were not silently changed or tuned against confirmation in this comparison.

TechnicalScore is `0.5 * Hit@10 + 0.3 * MRR + 0.2 * efficiency`; tokens are reported separately, not rewarded by the formula. Priors/checks have zero direct model tokens. Admission and dialogue changes can alter downstream token totals.

## Validation and provenance

- Full tests: 650 passed; lint and whitespace checks passed.
- Authored private-like capabilities: 19/19 assertions, zero API errors or fallbacks. This pack has no comparable TechnicalScore.
- Protocol and freeze: [REVIEW_PRIOR_PROTOCOL.md](REVIEW_PRIOR_PROTOCOL.md).
- Aggregates: [review-prior-results.json](review-prior-results.json).
- Raw reports: `runs/review-prior-*/report.json`; confirmation trace audit: `runs/review-prior-confirmation-audit-20260831/`.
- Production configuration equals the frozen `review_prior_mixed_both.json` configuration.
- Historical candidate configurations are preserved; their tests reference the frozen no-prior control.

These are consumed public development and locally synthetic sets, not the organizer's private set. Cycle 5 matches public popularity marginals, and the original branch's proposed 0.30 weight was already public-tuned. We fixed mechanisms before testing this corrected implementation, chose on screening, froze before other-set outcomes, and did not retune on confirmation or validation.
