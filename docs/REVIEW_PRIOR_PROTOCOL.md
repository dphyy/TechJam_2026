> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Review prior and constraint-stage comparison

Registered before running this implementation's candidate evaluations, 31 August 2026.
Main baseline: `5362742`. The feature branch's 0.30 proposal is already public-tuned;
this is a development comparison, not a claim of untouched public validation.

## Fixed mechanisms

- Count signal: `min(1, log1p(count) / log1p(500000))`.
- Raw stars: `(rating - 3) / 2`, with missing/invalid ratings neutral.
- Confidence-adjusted stars: raw stars multiplied by `count / (count + 20)`.
- Mixed: equal parts count and confidence-adjusted stars.
- Admission weight: 0.30, once before the 120 cutoff. No weight search.
- Optional final weight: 0.02, only after actual neural score replacement.
- Bonuses are replaced, never accumulated; neural fusion drops the old bonus marker.
- Existing guarded separation is protected even for very small score gaps.
- Invalid counts/ratings remain neutral; very large counts are saturated safely.
- Neural model, weight 0.75, top-30 budget, dialogue, and paging remain fixed.

## Evaluation order

1. Cycle 5 screening (160): control, count, raw stars, adjusted stars, mixed.
   Pick the highest TechnicalScore with no hit-rate regression and no runtime failures.
2. On the same development screening pack, compare the winning signal's admission-only,
   final-only, and both-stage settings. Prefer admission-only on a tie. Separately ablate
   the pre-constraint check, post-constraint check, and both checks for diagnosis only.
   Constraint removal cannot be promoted if it admits avoidable observed contradictions.
3. Freeze the candidate before public/confirmation outcomes. Compare control and candidate
   on public 200, Cycle 5 confirmation 80, Cycle 5 validation 80, and Cycle 3 screening
   160 (a low-popularity counter-distribution). Do not retune against these outcomes.
   Also run the authored private-like capability pack. All are local synthetic or
   consumed development evidence, not organizer-private results. Existing split
   consumption histories are retained; no reused split is described as pristine.
4. Promote only with no public or Cycle 5 confirmation/validation TechnicalScore loss,
   no hit-rate loss on those sets, no runtime failures, and disclose any counter-distribution
   regression. If a gate fails, retain the implementation disabled rather than adapting
   weights to the held-out results. Both constraint checks remain the production default.

Record tokens, latency, retrieval/admission diagnostics, constraint-stage reordering,
observed returned contradictions, and per-session results. Constraint checks and priors
are local CPU arithmetic/evidence checks: their direct model-token cost is zero.
Changed admission or session length can change downstream total tokens.

## Candidate freeze after screening

Before public/confirmation/validation runs: freeze `review_prior_mixed_both.json`.
Mixed admission-only scored 0.832220, count 0.830069, adjusted stars 0.826132,
raw stars 0.809154, control 0.814662. Mixed final-only scored 0.826065 and mixed
both-stage scored 0.837746 with unchanged 0.968750 HitRate. No weights were retuned.
The diagnostic constraint ablations use this same frozen mixed both-stage policy.
