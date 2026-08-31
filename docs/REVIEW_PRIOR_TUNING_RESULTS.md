# Fresh review-prior ratio and stage-weight search

Recorded 31 August 2026. **Decision: keep the production 50:50 blend at admission
weight 0.30 and post-neural weight 0.02.** The best development alternative was
count-only with the same weights. It improved the new reserved score, but did not
pass the practical development-gain requirement registered before evaluation.
The acceptance rule was not relaxed after seeing reserve outcomes.

The preceding implementation was committed and pushed first as `80c5e5b`
(`Add bounded review priors and verify ranking trade-offs`). This follow-up adds
an explicit `review_prior_count_fraction` control, defaulting to 0.50, without
changing the production configuration or its default behavior.

## Data discipline

This round did **not rerun or select against public, Cycle 5, or previous
confirmation/validation results**. Inventory used target identities only to exclude
prior exposure. It conservatively covered 43 target-pack files, excluding 3,139
target products and 3,136 loose-title families, including previously reserved packs.

A new hash-locked pack contains 160 development and 160 reserved sessions, with
zero target/title-family overlap between splits or with the inventory. Each split
has 64 buying, 64 browsing, 24 override and 8 boundary sessions, using the unchanged
simulator. Each has 40 targets in each review-count band: [0,5), [5,100),
[100,1000), [1000,5000).

**There were no unused title families with 5,000+ reviews.** This is new
lower-/medium-popularity proxy evidence, not a representative organizer-private
test or a replacement for high-popularity evidence. Absolute scores should not be
compared with public or Cycle 5. Title-family independence is heuristic; simulator
templates and the retrieval catalog are shared.

Protocol: [REVIEW_PRIOR_TUNING_PROTOCOL.md](REVIEW_PRIOR_TUNING_PROTOCOL.md).
Catalog, inventory, split files, baseline config, protocol, runtime, evaluator and
experiment source hashes were locked before development evaluation. One finalist
was frozen at `2026-08-31T03:49:35.835890+00:00`; reserved consumption was recorded
before loading its sessions. Both reserved runs finished by
`2026-08-31T03:53:49.857182+00:00`. No other candidate was evaluated on reserve.

## Development-only search

The full predeclared 45-point grid tested five count fractions (0, .25, .50, .75,
1), three admission weights (.10, .20, .30), and three post-neural weights
(0, .01, .02). The remainder of each mix is confidence-adjusted star quality,
not unadjusted average stars. Both constraint checks and every non-prior setting
were fixed. This is a bounded grid search, not a claim of a global optimum.

Best development result for each ratio, allowing its own stage weights:

| Count : adjusted stars | Admission | Post-neural | TechnicalScore | Hits / 160 |
|---|---:|---:|---:|---:|
| 0 : 100 | 0.20 | 0.02 | 0.823670 | 154 |
| 25 : 75 | 0.20 | 0.02 | 0.823907 | 154 |
| 50 : 50 (production) | 0.30 | 0.02 | 0.826080 | 154 |
| 75 : 25 | 0.30 | 0 | 0.827668 | 154 |
| 100 : 0 (frozen finalist) | 0.30 | 0.02 | 0.827701 | 154 |

Count-only leads the best 75:25 setting by just 0.000033. Its gain over production
is +0.001621, below the registered +0.005 minimum. The descriptive development
paired 95% interval is [-0.011089, +0.013721]; it is not selection-adjusted evidence
after a 45-point search. Development net hit count is unchanged, but one previously
found target is lost and one previously missed target is recovered.

Stage weights matter differently at different ratios. For count-only at admission
0.30, post weight 0.02 scores 0.827701 versus 0.824988 with no post prior. In
contrast, the best 75:25 configuration uses no post prior. The search does not
establish that repeated use always helps. The two numerical roles remain distinct:
admission before truncation, then optional final ordering after neural score
replacement, with the earlier adjustment removed rather than accumulated.

## Uncached matched results

| Set / setting | TechnicalScore | Hit@10 | MRR | MTTC | Model tokens |
|---|---:|---:|---:|---:|---:|
| Development / production | 0.826080 | 0.962500 | 0.602765 | 2.800000 | 1,758,341 |
| Development / count-only | 0.827701 | 0.962500 | 0.606920 | 2.781250 | 1,742,821 |
| Reserved / production | 0.795890 | 0.918750 | 0.605050 | 3.250000 | 1,876,969 |
| Reserved / count-only | 0.805644 | 0.918750 | 0.639231 | 3.275000 | 1,868,661 |

Reserved gain: **+0.009754**, paired 10,000-resample bootstrap 95% interval
**[+0.002213, +0.018150]**. Both find exactly the same 147/160 targets. Better MRR
drives the gain; average completion turns worsen slightly. Tokens fall by 8,308
(0.44%). Tokens are reported separately, not rewarded by TechnicalScore.

Reserved breakdowns are descriptive, not a basis for another selection:

| Review count | n | Production score | Count-only score | Delta |
|---|---:|---:|---:|---:|
| 0–4 | 40 | 0.824527 | 0.826527 | +0.002000 |
| 5–99 | 40 | 0.722322 | 0.732938 | +0.010616 |
| 100–999 | 40 | 0.826830 | 0.830875 | +0.004045 |
| 1,000–4,999 | 40 | 0.809881 | 0.832238 | +0.022357 |

| Scenario | n | Production score | Count-only score |
|---|---:|---:|---:|
| Buying | 64 | 0.815225 | 0.822016 |
| Browsing | 64 | 0.799103 | 0.813730 |
| Intent override | 24 | 0.842604 | 0.850521 |
| Boundary | 8 | 0.475357 | 0.475357 |

The small boundary group remains weak and unchanged. Do not infer reliable
subgroup effects from these sample sizes.

## Decision and correctness limits

| Registered requirement | Result |
|---|---|
| Development gain at least +0.005 | Fail: +0.001621 |
| Reserved gain at least +0.005 | Pass: +0.009754 |
| No net hit-rate loss or more marked contradictions | Pass |
| Reserved paired 95% interval strictly above zero | Pass |
| Reserved tokens at most 5% above baseline | Pass: 0.44% lower |
| No runtime fallbacks/errors | Pass |

Overall: **not promoted**. Keep `configs/selected.json` unchanged. The reserved
result is encouraging evidence for count-only on fresh lower-/medium-popularity
targets, not permission to lower the development threshold or retune on reserve.
New high-popularity data would be needed to address the missing distribution;
these now-consumed sessions cannot serve as another untouched confirmation set.

Both constraint guards remained on. All grid runs and both reserved runs had no
runtime fallbacks/errors. Development had zero state-marked returned contradictions;
reserved had **28 in both configurations**. Thus the finalist does not worsen this
metric, but neither configuration guarantees contradiction-free recommendations.
The guards demote contradictions rather than strictly removing them, and parsed
state itself can be wrong. No parser or fallback-slate changes were made using
reserved outcomes. The existing limitations are documented in
[REVIEW_PRIOR_RESULTS.md](REVIEW_PRIOR_RESULTS.md).

## Verification and artifacts

- 661 unit tests pass; lint and whitespace checks pass.
- Development memoization reused only fixed-model query/document logits. It
  charged every logical input token: 78,327,467 across the 45 grid runs. Physical
  grid prediction work was 1,912,453 tokens / 11,126 pairs, with 455,514 pair hits.
  These are local-model experiment counts, not external API usage.
- The control and finalist reproduced identical full session outcomes and token
  totals uncached. Reserved runs were uncached. Cached timings are not production
  latency measurements.
- Runtime/evaluator/config/data hashes remained unchanged from lock through both
  phases. Evidence writes are exclusive; reserved reopening is rejected before
  loading data or constructing an agent.
- Full target-free grid, metrics, breakdowns and hashes:
  [review-prior-tuning-results.json](review-prior-tuning-results.json).
- Frozen experimental config, **not selected**:
  [review_prior_tuned_count_frozen.json](../configs/review_prior_tuned_count_frozen.json).
- Runner: [review_prior_tuning.py](../experiments/review_prior_tuning.py).
- Local raw evidence: `runs/review-prior-tuning-v2/`; locked data:
  `artifacts/review-prior-tuning-v2/`. Both directories remain ignored by Git.

`python -m experiments.review_prior_tuning verify` audits the existing lock.
Do not rerun `reserve`: its consumption ledger intentionally refuses another run.
