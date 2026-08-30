# Cycle 5 registration: popularity-matched synthetic target pool

Registered 29 August 2026, before the pack was generated and before any arm was
run against it. This registers a *measurement apparatus* change, not a system
change. No agent, evaluator, model, or configuration is modified by this work.

## Problem

The Cycle 3 synthetic pack draws targets uniformly over eligible loose title
families. The organizer does not draw uniformly. Their sessions are anchored on
the final purchased record of a user surviving a 5-core leave-last-out join, so
the target pool is popularity-weighted: to be somebody's last purchase, a
product has to appear in purchase histories at all, and popular products
dominate histories.

Measured against the 200 released-public targets, which are a known sample of
that pool:

| Property of the target product | Released public 200 | Cycle 3 screening 160 |
|---|---:|---:|
| `rating_number` median | 6,846 | 14 |
| `rating_number` p25 | 894 | — |
| Targets under 25 ratings | 2% | 61% |
| Targets with a price | 89% | 16% |

The Cycle 3 pack is also measurably *easier* than the released public set: the
same recorded control scores `0.811786` / `0.925000` Hit@10 on Cycle 3 screening
against `0.786724` / `0.895000` on public.

## What this does and does not claim

It does **not** claim that Cycle 3 rejected a good candidate. That hypothesis was
tested and is not supported: intent cards built from both populations are
near-identical, because the card builder consumes only the first few cleaned
feature/detail strings and truncates to four constraints.

| Intent-card property | Public 200 | Cycle 3 screening 160 |
|---|---:|---:|
| Distinct constraints per card | 4.00 | 3.94 |
| Constraint length, median chars | 14 | 14 |
| Constraints that are a bare material/color word | 25% | 23% |

The claim is narrower and is about calibration only: a proxy holdout used to
accept or reject candidates should be drawn from the population the system will
be graded on. The organizer states that public and private evaluation sessions
use separate users and target products, drawn from the same frozen catalog by
the same construction. The released public 200 is therefore the only available
sample of the private population, and matching its observable marginals is the
best available estimator.

No Cycle 3 result is retracted, and no Cycle 3 artifact is modified or deleted.

## Selection rule

Identical to Cycle 3 in every respect except which families are drawn:

- eligibility, loose-title family construction, one hash-chosen member per
  family, split-local hash ordering, and the fixed scenario mix are unchanged;
- released-public targets and families are excluded, exactly as before;
- **all three Cycle 3 splits are additionally passed as consumed datasets**, so
  the new pack shares no target and no loose title family with them.

The one new step is a stratified draw. Eligible families are bucketed by the
`rating_number` of their chosen member into six popularity bands. Per-band
quotas are *derived at runtime* from the released-public 200 by largest-remainder
rounding, not hardcoded. Bands are filled from the highest downward; any
shortfall in a band cascades into the next lower band and is recorded.

Bands: `[0, 5) [5, 100) [100, 1000) [1000, 5000) [5000, 20000) [20000, inf)`.

Within each band, members are dealt across splits on a repeating
`screening, screening, confirmation, validation` cycle, so every split receives a
proportional share of every band rather than screening absorbing the popular
tail.

## Pre-declared feasibility limitation

The catalog cannot reproduce the extreme upper tail. Measured before generation:

| Band | Public share | Needed of 320 | Eligible families |
|---|---:|---:|---:|
| 0 – 5 | 1.0% | 3 | 15,077 |
| 5 – 100 | 4.0% | 13 | 24,909 |
| 100 – 1,000 | 20.5% | 66 | 7,353 |
| 1,000 – 5,000 | 20.5% | 66 | 1,039 |
| 5,000 – 20,000 | 30.0% | 96 | 193 |
| 20,000 + | 24.0% | 77 | **38** |

The top band is short by roughly 39 targets, which will cascade into the
5,000–20,000 band. The combined `>= 5000` share is therefore expected to be
preserved at about 54%, matching public, while the split *within* that region is
compressed toward the lower half. This is a known and accepted distortion,
recorded here before generation rather than discovered afterwards. It is far
smaller than the distortion it replaces.

The realised per-band counts and shortfalls are written into the pack manifest
audit, so the achieved distribution is verifiable without rerunning the draw.

## Use

Splits are used exactly once each, in order, under the existing gate discipline:
screening, then confirmation, then validation. The Cycle 3 pack remains valid for
already-recorded comparisons; it is not re-run and not deleted. Any arm measured
against Cycle 5 must state which pack it used, because the two are not
comparable to each other.

No promotion, rejection, or selection decision is authorized by this document.
