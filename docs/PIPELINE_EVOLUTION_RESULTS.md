> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Pipeline Evolution Results

This document records the implementation and evaluation of [the pipeline evolution plan](../plan.md). All public-set measurements are development evidence. The public set has already been used for development and is not a fresh holdout or evidence of private-test performance.

Phases 9-14 were implemented later as independently gated robustness work and are
recorded separately in [the roadmap results](ROADMAP_IMPLEMENTATION_RESULTS.md).

## Outcome

Phases 0 through 7 are implemented. Behavior-changing features are independently configurable and remain disabled unless they passed the promotion rule. No candidate achieved the required `>= 0.01` TechnicalScore improvement, so `configs/selected.json` remains unchanged.

After the main merge, varied non-repeating question prompts, and soft-price handling, the selected configuration reproduces:

| HitRate@10 | MRR | MTTC | TechnicalScore | Fallback turns |
|---:|---:|---:|---:|---:|
| 0.895000 | 0.613746 | 3.245000 | 0.786724 | 0 |

It also passes all 19 assertions in the authored private-like capability pack with zero failures, unverified assertions, API errors, or fallback turns. See [the merge decisions](MERGE_DECISIONS.md) for the rejected strict one-question result and the release/experiment boundary.

## Phase Ledger

| Phase | Commit | Configuration | TechnicalScore | Decision |
|---|---|---|---:|---|
| Plan | `59b9251` | n/a | n/a | Recorded implementation and promotion gates |
| 0-1 diagnostics and intent | `bf89ee1` | selected behavior | 0.786724 | Kept; diagnostic-only |
| 2 typed retrieval plan | `d7da3b7` | selected behavior | 0.786724 final frozen replay | Kept; scope parsing gated |
| 3 routed retrieval | `5eac9c9` | `routed_retrieval.json` | 0.785165 | Rejected for promotion |
| 4 compatibility guard | `0df987f` | `product_guard.json` | 0.772674 | Rejected for promotion |
| 5 structured reranking | `24cc8bd` | `structured_rerank.json` | 0.754929 | Rejected for promotion |
| 6 intent clarification | `0c0fd33` | `intent_clarification.json` | 0.753090 | Rejected for promotion |
| 7 runtime adaptation | `70ab315` | `runtime_adaptation.json` | 0.786636 | Rejected for promotion |

The reference for promotion is TechnicalScore `0.786724`. Scores above were produced with the same 200-row public development set and 50,000-product catalog. Generated raw reports remain ignored local artifacts.

## What Was Added

- An explainable `buying`, `browsing`, and `mixed` intent decision with specificity, confidence, over-generality, and target-independent reasons.
- A typed retrieval plan preserving object terms, positive and negative evidence, hard and soft intent, use case, alternatives, source turn, and optional component scope.
- Conditional sparse/dense route orchestration with buying, browsing, and mixed weights, dynamic candidate budgets, route-overlap diagnostics, and sparse fallback.
- Conservative object/accessory/component classification and scope-aware evidence. Unknown types and missing metadata are not contradictions.
- Optional labeled cross-encoder context with deterministic field priorities and the original post-rank guard.
- Intent-aware clarification with explicit turn cost, no-preference suppression, bounded fallback, final-turn protection, and an optional broad-query rerank cutoff.
- Conservative profile distillation and inferred-soft-signal decay with source provenance. Explicit hard constraints and negative feedback never decay.
- Stage diagnostics for route weights/counts, Recall@30/@60/@120, guard/ranker counts, question reasons, component latency, tokens, and fallbacks.

## Interpretation By Pillar

1. Intent routing and hybrid retrieval are implemented and ablatable. Routed sparse retrieval improved browsing HitRate@10 from `0.9125` to `0.9250`, but reduced buying performance and did not improve the aggregate score. Dense routing is implemented but was not evaluated with prepared dense assets in this cycle.
2. Multi-turn state remains the strongest behavioral pillar. Intent-aware questions are available, but the official simulator strongly preferred the selected bounded `other` policy; the natural policy increased MTTC to `4.645`.
3. Dynamic context programming is implemented as deterministic session-level orchestration, not self-modifying code. Generic profile tags produced effectively neutral target performance while increasing reranking work through turn-sensitive cache invalidation.
4. Evaluation is substantially stronger: end-to-end score is supplemented by router, state, retrieval-depth, ranking, constraint, product-role, dialogue, latency, token, and fallback diagnostics.

## Why Phase 8 Was Not Activated

Phase 8 is optional in the plan and requires Phases 1-7 to justify additional model complexity. They did not produce a promotable gain. A hosted or local generative LLM was therefore not added merely to satisfy the phrase "LLM semantic ranking." Mercury continues to use the pinned local MiniLM cross-encoder for selected semantic ranking, with deterministic state and guards retaining final authority.

## Final Verification

```text
419 tests passed
Ruff passed
pip check passed
private-like: 19 passed, 0 failed, 0 unverified
selected public: 0.786724, 0 fallback turns
```
