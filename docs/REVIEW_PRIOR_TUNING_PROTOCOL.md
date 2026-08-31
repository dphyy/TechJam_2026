# Fresh review-prior ratio and stage-weight tuning

Registered 31 August 2026, before generating the new pack or evaluating this grid.
Production baseline: `80c5e5b`, mixed count fraction 0.50, pre weight 0.30,
post weight 0.02. The preceding implementation was committed and pushed first.

## Data boundary

Inventory all target JSONL packs under data/artifacts, plus dataset paths in
existing run manifests/reports. Exclude every known target and loose-title family,
including previously reserved packs conservatively. Inventory uses target identity
only, not their outcomes. Existing packs and consumption ledgers are not modified.
Generate a new immutable pack with seed `review-prior-tuning-20260831-v2`:

- 160 development and 160 reserved sessions; one hash-chosen product per loose
  title family, zero target/family overlap between splits and against the inventory.
- Equal counts from four review-count bands: [0,5), [5,100), [100,1000), [1000,5000).
  Within each band, hash-order representatives and alternate assignment to the splits.
- Independently hash-order each split before assigning 64 buying, 64 browsing,
  24 override and 8 boundary scenarios through the unchanged official simulator.
- Lock catalog, inventory-file, dataset, script, protocol, baseline config and
  evaluator hashes before development runs. Record reserved consumption before its
  first evaluation and refuse a second opening.

The inventory audit found **no unused family with 5000+ reviews** after conservative
exclusions. This round therefore measures fresh lower-/medium-popularity products,
not the organizer's high-popularity target distribution. It is not comparable in
absolute score to Cycle 5 or public. Family grouping is a title heuristic, not a
manufacturer-family guarantee; dialogue templates remain the same simulator.

## Fixed search

Full factorial, 45 configurations, no post-outcome grid expansion:

- Count fraction: 0, 0.25, 0.50, 0.75, 1 (the remainder is confidence-adjusted stars).
- Admission weight: 0.10, 0.20, 0.30.
- Post-neural weight: 0, 0.01, 0.02.

The current production setting is one grid point. Both constraint checks, model,
neural weight 0.75, 30-pair budget, state, questions, and paging remain unchanged.
Public and all old confirmation/validation outcomes are prohibited for selection.

Development-only memoization may reuse identical query/document logits under a
fixed model. Cache keys contain no sample IDs or targets. Every logical call still
counts its full model input tokens; physical cache savings are reported separately,
and cached latency is not a production benchmark. Reproduce the uncached control
and the chosen finalist on development, requiring identical session results and
logical tokens. On parity failure, stop rather than use unverified cached scores.

Choose the best healthy non-control grid point with no development hit-rate loss
and no increase in state-marked returned contradictions. If none is eligible,
freeze the best healthy non-control point only for the requested diagnostic reserved
comparison and explicitly record development-gate failure. Sort by TechnicalScore,
then lower post weight, lower pre weight, ratio closest to 0.5, then ratio.
Freeze exactly one candidate config, source hashes and development report digest
before opening reserved outcomes. No second finalist or retuning against reserved.

## Reserved decision

Evaluate the frozen baseline and finalist once each, uncached, on the same new
160-session reserved pack. Report paired 10,000-resample bootstrap intervals and
review-band/scenario breakdowns. A fresh-proxy acceptance requires:

- development and reserved TechnicalScore gains each at least 0.005;
- no Hit@10 loss, no more state-marked contradictions, no runtime fallbacks/errors;
- reserved paired 95% TechnicalScore-delta interval strictly above zero;
- reserved total model tokens at most 5% above baseline.

If any gate fails, keep the pushed production setting. Even a pass establishes only
fresh-proxy evidence: do not silently replace the production setting or claim a new
public score without the missing high-popularity generalization evidence/user direction.
The request authorizes this tuning round, not repeated reserve reuse or a public sweep.

Implementation: `experiments/review_prior_tuning.py`; local lock and outputs under
`artifacts/review-prior-tuning-v2` and `runs/review-prior-tuning-v2` (ignored).
