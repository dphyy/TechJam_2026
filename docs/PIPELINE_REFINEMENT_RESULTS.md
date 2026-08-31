# Merged pipeline refinement results

Recorded 30 August 2026 after merging `origin/main` into
`feat/improve-generalization`. The merge composes the generalization branch's
typed intent and bounded-compute pipeline with main's parser repairs, evidence
experiments, reranker selection, turn-budget fallback, and slate paging.

## Selection outcome

The selected runtime remains the reliable sparse/D30 MiniLM pipeline. A later
registered follow-up advances an unchanged ranking from the first repeat and
resets to page 1 whenever the runtime classifier detects an intent override.
This supersedes the original turn-5 paging boundary while preserving its
eligibility protection. Intent classification
uses dataset-tuned interpretable weights for diagnostics, but intent-routed
retrieval remains disabled because it did not improve downstream output.

The selected pipeline is:

```text
reversible grouped-alternative ledger
  -> broad + category-scoped FTS5 retrieval
  -> unknown-safe hard-contradiction guard
  -> retain 120 candidates
  -> MiniLM rerank of the first 30 at weight 0.75
  -> guard again
  -> simple non-repeating `other` questions, capped at four
  -> Top 10; advance on each unchanged ranking, reset to page 1 on intent override
```

Dense retrieval, routed retrieval, positive evidence boosts, role/composition
boosts, source-alias retrieval, product compatibility demotion, D60 cascade,
multi-hypothesis retrieval, and semantic question gating remain disabled.

## Intent-rule calibration

The original buying/browsing rule weights were handwritten. Their replacement
was selected without evaluator targets or the consumed intent sealed fold:

1. Use only the 84-row independently authored training split.
2. Keep paraphrase/author/intent-card/product-family groups together.
3. Form five deterministic label-stratified group folds.
4. Run bounded coordinate search on a 0.05 grid.
5. Use the 18-row validation split only as acceptance evidence.

| Metric | Legacy rules | Tuned rules |
|---|---:|---:|
| Five-fold train macro F1 | 0.533189 | 0.788046 |
| Whole-train macro F1 | 0.560064 | 0.780087 |
| Validation macro F1 | 0.488807 | 0.599845 |
| Validation buying recall | 1.000000 | 0.833333 |
| Validation browsing recall | 0.000000 | 0.666667 |
| Validation mixed recall | 0.500000 | 0.333333 |

The selected coefficients are explicit in `configs/selected.json`. A hard
constraint receives zero intent-mode weight because it did not distinguish
buying from mixed requests in grouped training; hard constraints still retain
their full deterministic authority in state and ranking. The weaker mixed recall
is why the classifier is diagnostic only and does not control final retrieval.

`python -m experiments.tune_intent_rules --output PATH` reproduces the search
and refuses to read the sealed split.

## Downstream selection evidence

The public-target-excluding development pack contains 80 sessions. The final
pack contains 40 target/user-disjoint sessions and was opened only after paging
cleared the development and correctness gates.

| Split / arm | Hit@10 | MRR | MTTC | TechnicalScore | Tokens |
|---|---:|---:|---:|---:|---:|
| Unseen development, fixed slate | 0.887500 | 0.636781 | 3.300000 | 0.788784 | 874,212 |
| Unseen development, paging | 0.912500 | 0.651066 | 3.212500 | 0.807320 | 874,212 |
| Final control, fixed slate | 0.950000 | 0.655119 | 2.675000 | 0.838036 | 423,141 |
| Final candidate, paging | 0.975000 | 0.658244 | 2.650000 | 0.851973 | 423,141 |
| Consumed public, paging (descriptive) | 0.970000 | 0.645919 | 2.980000 | 0.839176 | 2,375,008 |
| Consumed public, early paging + override reset | 0.970000 | 0.641633 | 2.905000 | 0.839390 | 2,372,103 |
| Consumed public, stable-head unseen paging + semantic reset | 0.980000 | 0.640647 | 2.860000 | 0.844994 | 2,372,103 |
| Consumed public, bounded mixed review prior | 0.990000 | 0.683974 | 2.670000 | 0.866792 | 2,370,993 |

Paging improves TechnicalScore by `+0.018536` on unseen development and
`+0.013937` on the final split, with `+0.025` Hit@10 on both. It performs no
extra retrieval or inference and uses identical model tokens. The final run had
zero fallbacks and unchanged source throughout. The authored private-like pack
also passed 19/19 assertions with zero API errors or fallbacks.

The guarded early-paging follow-up is descriptive public-development evidence.
Against a fresh same-source turn-5 control it changes TechnicalScore by
`+0.000214`, preserves HitRate, reduces MTTC by `0.075`, and lowers MRR by
`0.004286`. It is selected under the owner's TechnicalScore non-decline rule;
the component trade-off remains explicit.

## Strategy trade-offs

| Strategy | Evidence | Decision and reason |
|---|---|---|
| Fixed D30 + paging | Positive on two target/user-disjoint splits; no token cost | Selected. It exposes already-computed candidates only after an unchanged slate has failed. |
| Intent-routed sparse retrieval | Three scenario-route weights all scored exactly 0.807320, matching paging | Disabled. It changed retrieval work but not recommendations, added 6.3k-7.3k tokens, and raised p95 as high as 0.428s. |
| D30-to-D60 uncertainty cascade | Earlier unseen delta +0.000313 | Disabled. Twice the reranking work did not deliver a practical gain. |
| Two intent hypotheses | Earlier unseen delta -0.020354 | Disabled. Dividing a fixed candidate budget weakened the strongest lexical route. |
| Semantic value-gated questions | Earlier unseen delta -0.150318 | Disabled. Fewer model pairs could not compensate for slower disclosure and MTTC. |
| Structured reranker context | Public delta -0.031795 in the earlier cycle | Rejected. Labels changed the cross-encoder input distribution and harmed ranking. |
| BGE reranker | Public +0.003106, lower MRR, 6.6x p95 | Rejected. More hits did not offset poorer top-rank ordering and latency. |
| Role/composition/source-alias evidence | Correct or active, but no practical score gate | Kept as isolated capability configs, not combined into release. Combining weak arms after seeing outcomes would overfit consumed data. |

The main generalization lesson is that broader or more semantic routing is not
automatically better. The selected lexical route already has high target recall;
splitting candidate budgets or changing cross-encoder inputs mainly perturbs good
ordering. Paging wins because it addresses a policy impossibility—re-serving an
unchanged failed slate cannot score—without weakening retrieval or ranking.

These are local public/synthetic measurements, not organizer-private results.
The final split is now consumed and must not be reused as held-out evidence for a
future candidate.
