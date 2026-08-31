> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Exact pair-logit cache protocol

Registered on 30 August 2026 before Phase 1A runtime implementation or cache
benchmarking.

## Hypothesis and fixed candidate

The pinned MiniLM cross-encoder is deterministic for an exact effective query
and serialized product document. Reusing that exact logit across sessions can
remove repeated local inference without changing candidate scores, ordering,
page membership, questions, or recommendation IDs.

The fixed candidate adds an opt-in, process-local LRU cache with capacity 8,192
pair logits. The cache is thread-safe and belongs to one `NeuralRanker` instance;
session reset does not clear it. A key contains:

- reranker kind and pinned model revision;
- serializer version and document mode;
- device, CPU-thread count, and inference batch size;
- SHA-256 of the complete effective query;
- product ID and SHA-256 of the exact serialized document;
- maximum sequence length.

Only exact key matches may reuse a score. Cache misses are restored to their
original candidate positions after model inference. The candidate exposes
capacity, size, hits, misses, evictions, and actually evaluated pairs. Any model
or score failure retains the existing sparse fallback behavior. No cache data is
persisted, and no target, sample, scenario, future-turn, or evaluator data enters
the key or value.

Phase 1B identical-document grouping is not part of this candidate. Distinct
product IDs remain distinct keys even when their documents happen to match.

## Fixed evidence

Because this is behavior-preserving work, selection uses exact paired response
parity rather than target-score improvement. Compare source-matched control and
candidate configurations on:

- the 200-session consumed public development set;
- the frozen 80-session unseen development proxy;
- the existing private-like and metamorphic capability suites;
- a reusable-pair workload that repeats the same ordinary query in independent
  sessions, so the agent-level per-session cache cannot create the reuse.

The reusable workload is fixed at 40 independent sessions, one turn per session,
with the message `blue cotton shirt`, empty profiles, and `top_k=10`. Control and
candidate use the same catalog, source, model, and selected ranking settings.

## Gates

Promotion requires every condition below:

- byte-equivalent messages, questions, recommendation IDs/order, and completion
  usage for every paired turn; `usage.prompt_tokens` may only decrease because
  it reports actually evaluated local-model input;
- identical aggregate HitRate@10, MRR, MTTC, TechnicalScore, questions, and
  recommendation ordering;
- zero new fallbacks, invalid IDs, cache-provenance failures, or correctness-test
  regressions;
- at least 20% fewer actually evaluated neural pairs on the reusable workload;
- either at least 15% lower reusable-workload warm p95 or the evaluated-pair gate;
- no more than 16 MiB additional peak RSS at the fixed 8,192-entry capacity;
- deterministic LRU eviction and exact invalidation when query, product ID,
  document content/version/mode, model kind/revision, or maximum length changes.

If all gates pass, only `neural_logit_cache=true` and
`neural_logit_cache_size=8192` may be added to `configs/selected.json`. Phase 1B
must remain a separate later change.

## Result

The candidate passed response parity and correctness but failed the registered
memory gate and was not promoted.

| Workload | Control pairs | Cache pairs | Pair reduction | Control p95 | Cache p95 | Semantic parity |
|---|---:|---:|---:|---:|---:|---|
| 40-session exact-reuse | 1,200 | 30 | 97.5% | 0.206642s | 0.017893s | Exact |
| Public development | 13,710 | 12,460 | 9.1% | 0.368823s | 0.509760s | Exact |
| Unseen development | 5,460 | 5,070 | 7.1% | 0.419916s | 0.528236s | Exact |

Public and unseen HitRate, MRR, MTTC, TechnicalScore, questions, messages, and
recommendation ordering were identical turn-for-turn. Public prompt tokens fell
from 2,372,103 to 2,213,531 and unseen tokens fell from 874,155 to 831,949. No
fallbacks or agent errors occurred, and all private-like capability assertions
passed.

Peak RSS increased by about 9.8 MiB on public but 80.1 MiB on the unseen proxy,
exceeding the fixed 16 MiB gate. Ordinary-workload p95 also regressed in both
measurements despite the exact-reuse win. The selected configuration therefore
remains unchanged. The implementation and `configs/neural_logit_cache.json`
remain opt-in evidence for later cache-layout research; they are not a release
claim.
