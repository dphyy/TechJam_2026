> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Continuation matrix and causal attribution

## Decision

Phase 7 is complete. The second source-independent matrix contains 480 open
training rows, 160 sealed screening rows, and 80 sealed confirmation rows. The
original robustness-matrix final split was used only as an exclusion source and
was not evaluated; its consumption ledger remains sealed.

The lock excludes every target, loose-title family, and deepest category group
represented in the supplied public, frontier, page-local, unseen, and first
robustness-matrix sources. It also checks the new author, user, dialogue-template,
paraphrase, and wording-family namespaces against the first matrix annotations.

## Frozen evidence

- Seed: `robustness-matrix-20260830-v2`
- Training: 480 rows, SHA-256
  `072a23723490812be79d5dc437cce3209e91c98e7b1b489278a648b07eee0468`
- Screening: 160 rows, SHA-256
  `325a5f3de837b364855de500ca9638e6332ca2c287916cef36641c0289eaef18`
- Confirmation: 80 rows, SHA-256
  `0811fac13c886751af72364e45066f37bd81394424b54a89fbd099cdfd8766a4`
- Cross-split target and group overlap: zero
- Previously represented category groups excluded: 368
- Eligible new category groups: 404

The manifest records a conservative power receipt. At a 0.90 baseline, a five
percentage-point effect, and a 75% effective-sample factor, the approximate
power is 0.447 for screening and 0.252 for confirmation. These relatively low
values make the no-loss paired gate more informative than a significance claim.

## Runtime receipts

The agent now emits target-independent SHA-256 signatures for canonical active
state and retrieval plans. Stage memberships distinguish retrieval, pre-neural
guards, candidate limiting, admission order and selection, neural order,
post-neural guards, final ranking, control page, and returned page. Timing uses
non-overlapping parsing/state, intent/planning, retrieval, pre-neural ranking,
admission, neural, post-neural ranking, policy, optional page reranking, and
response-assembly buckets. Each neural call also reports per-product serialized
characters, populated fields, and exact tokenizer input length when evaluated.

`experiments/causal_attribution.py` joins these runtime-safe receipts to target
labels only after evaluation. It reports paired gains/losses and assigns misses
to runtime, state/intent, retrieval, admission, ranking, guard, question, or
paging stages. This module is not imported by runtime code.

## Open-training control

The selected release was run on the 480 open training rows:

| Metric | Result |
|---|---:|
| HitRate@10 | 0.979167 |
| MRR | 0.677852 |
| MTTC | 2.145833 |
| TechnicalScore | 0.870022 |
| Warm turn p95 | 0.533859s |
| Fallback turns | 0 |

Offline attribution identified four admission misses, one within-D30 ranking
miss, and five question/policy misses among the ten failures. All 1,020 turn
timings reconciled within the registered maximum of 20 ms or 10% of observed
turn latency. This evidence justifies independently testing Phase 10 admission,
Phase 11 ordering, and Phase 12 policy after Phase 8 state invariance; it does
not authorize opening screening.

## Reproduction

Generate the lock with `experiments.robustness_matrix_v2_prepare`, passing every
consumed dataset and the first matrix annotation files. Evaluate only the open
training rows with `experiments.run`, then run:

```bash
python -m experiments.causal_attribution \
  --dataset artifacts/robustness-matrix-v2/training.jsonl \
  --control-run runs/continuation-v2-training-control-20260830
```

Use the consumption-ledger command before any later screening or confirmation
run. Never use this attribution output as a runtime feature.
