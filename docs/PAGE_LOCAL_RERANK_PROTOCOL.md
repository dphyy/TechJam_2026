> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Page-local reranking protocol

Registered on 30 August 2026 before the page-local target pack was generated
and before page-local runtime behavior was implemented. This hypothesis was
derived from the consumed progressive-frontier screening result and must not be
selected on that pack.

## Hypothesis and fixed design

The rejected global frontier improved MRR but changed slate membership and lost
HitRate. The page-local candidate preserves the selected runtime's retrieval,
global ranking, page counter, and page membership. On an eligible continuation,
it may neural-score and reorder only the exact products already assigned to the
next control page.

The fixed candidate has these limits:

- `seen_aware_slate=false`
- `progressive_frontier_rerank=false`
- `page_local_rerank=true`
- at most 10 new pairs per batch
- at most 2 batches and 20 new pairs per session
- at most 0.25 seconds for a page-local scoring call before its result is
  discarded
- no run on the first page, an override, or an informative state update
- logits cached by session, state revision, and product ID
- any exception, malformed output, or over-budget result serves the original
  page in its original order and records `frontier_page_rerank`

Page membership must be identical to control. Only order within a page may
change. The runtime may use active state, candidates, page state, prior model
scores, and ordinary catalog evidence. It may not use sample IDs, targets,
scenario labels, future turns, or evaluator outcomes.

The same source also fixes an independent override edge case: a strong explicit
correction directive is an override even when no replacement value can be
parsed, unless protected by the existing no-change guard. This source-matched
fix is present in both evaluation arms.

## Locked data

Generate a new popularity-matched Cycle 5 pack with seed
`page-local-20260830-v1`. Exclude released-public targets and every previously
consumed target and loose title family, including all three progressive-frontier
splits. Screening, confirmation, and validation must be mutually disjoint.

Only two fixed arms may be compared on screening:

1. `control`: selected configuration on the new source.
2. `page-local`: selected configuration plus the fixed page-local settings.

No trigger, budget, batch, scoring, or ranking rule may be changed after viewing
screening outcomes. At most the page-local candidate may advance.

## Gates

Screening promotion requires every condition below against the source-matched
control:

- Technical Score gain of at least `0.005`.
- No HitRate@10 loss.
- Positive MRR improvement.
- No HitRate, MRR, or Technical Score loss on the intent-override slice.
- Zero agent errors, fallbacks, invalid IDs, source drift, or correctness-test
  regressions.
- Warm p95 turn latency no more than `1.20x` control.
- Total prompt tokens no more than `1.15x` control.

If screening passes, freeze source and configuration before one confirmation
run per arm. Confirmation requires non-negative overall and intent-override
Technical Score, HitRate, and MRR deltas, plus the same correctness and resource
gates. Validation remains unopened unless a later final-release protocol uses
it. The consumed public set may be run once after selection only as a descriptive
non-regression check and cannot rescue a failed candidate.

## Locked pack

The pack was generated and verified before page-local runtime implementation.

| Split | Sessions | SHA-256 |
|---|---:|---|
| Screening | 160 | `3267c31e786a1eb06958d23317d62d1c7384639759960e395fef3af73bf9567e` |
| Confirmation | 80 | `7a09fe27b697e1986884a45c311d1329e39097edf1df2009ad76381d1877e243` |
| Validation | 80 | `41d1f29fc99fbc5cbce2890246ef80d276bbb0111ad7e12eac11aa1e936493d4` |

## Screening result

Both arms completed with zero fallback turns and zero agent errors.

| Arm | Technical Score | HitRate@10 | MRR | MTTC | p95 seconds | Prompt tokens | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Control | 0.810155 | 0.925000 | 0.635933 | 3.15625 | 0.354375 | 1,822,332 | Baseline |
| Page-local | 0.812793 | 0.925000 | 0.644727 | 3.15625 | 0.454092 | 1,893,131 | Reject |

The page-local arm preserved every control hit and MTTC, improved MRR by
`0.008794`, and improved Technical Score by `0.002638`. The intent-override
slice was exactly unchanged. Prompt tokens increased by `1.0389x`, within the
registered ceiling, but p95 latency increased by `1.2814x`. The candidate
therefore failed both the minimum Technical Score gain and latency gates.

Across 493 turns, page-local reranking ran 38 times. The trace audit found zero
page-membership violations, zero duplicate returned IDs, no session above two
additional batches, and no session above 20 additional neural pairs. The
candidate did not advance; confirmation and validation remain unopened. The
implementation remains available through `configs/page_local_rerank.json`, and
the selected configuration remains unchanged.
