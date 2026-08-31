> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# MiniLM batch and CPU-thread benchmark protocol

Registered on 30 August 2026 before configurable neural batch-size runtime work
or benchmark execution.

## Fixed matrix

Benchmark every combination of:

- inference batch size: `16`, `30`, `32`;
- PyTorch CPU threads: `2`, `4`, `6`, `8`.

Use the pinned selected MiniLM, selected `head` serializer, CPU device, and the
same deterministic D30 pair set produced for the ordinary query
`blue cotton shirt`. Each process performs two unmeasured warmups followed by 20
measured scoring calls. It records the exact 30-logit vector digest, maximum
within-process logit drift, ranking digest, p50, p95, maximum latency, cold start,
prompt tokens, and peak RSS.

Each matrix cell runs in a fresh process. The benchmark may not inspect target
IDs, evaluator labels, simulator outcomes, or future turns. Its fixed product IDs
come only from ordinary selected retrieval for the registered query and are
benchmark inputs, not runtime rules.

## Gate

A setting may replace selected batch `16` / threads `4` only if:

- every measured logit is finite;
- repeated logits are exact or within `1e-6` absolute tolerance;
- final ordering and IDs exactly match the batch-16 / threads-4 reference;
- p95 improves by at least 15% in two source-matched repetitions or evaluated
  throughput improves without a p95 regression;
- peak RSS does not rise by more than 64 MiB;
- the complete correctness, private-like, and failure suites remain clean;
- a source-matched public descriptive run has exact shopper-visible parity and
  no fallback regression.

Only `threads` and `neural_batch_size` may differ in a promoted configuration.
Logit caching and document grouping remain disabled so their effects cannot be
confounded with this matrix.

## Result

All 12 cells completed with finite logits, zero within-cell drift, identical
candidate IDs, and the same final ranking. Batch 16 versus batch 30/32 differed
by at most `9.5367431640625e-7`, within the registered tolerance.

| Threads | Batch 16 p95 | Batch 30 p95 | Batch 32 p95 |
|---:|---:|---:|---:|
| 2 | 0.207154s | 0.200162s | 0.206699s |
| 4 | 0.190879s | **0.178213s** | 0.191913s |
| 6 | 0.184170s | 0.219615s | 0.248106s |
| 8 | 0.312646s | 0.392442s | 0.378537s |

The fastest p95 cell was four threads / batch 30, improving on selected four
threads / batch 16 by `6.63%`. It remained within the 64 MiB RSS cap but did not
reach the required 15% improvement. Six threads / batch 16 improved p95 by only
`3.51%`; eight threads regressed materially. No cell advanced to a second
repetition or public run, and selected remains batch 16 / four threads.
