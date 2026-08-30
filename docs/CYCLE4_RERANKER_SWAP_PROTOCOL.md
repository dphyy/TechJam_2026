# Cycle 4 reranker-swap registration

Registered 28 August 2026, before any candidate result was produced or inspected.
It follows the completed Cycle 3 depth work, where D60 and D120 both raised
released-public scores but failed fresh screening and were rejected.

## Single candidate

`cycle4_bge_base` changes exactly one configuration field, `reranker_model`, from
the pinned `cross-encoder/ms-marco-MiniLM-L6-v2` to the pinned
`BAAI/bge-reranker-base` (revision `2cfc18c9415c912f9d8155881c133215df768a70`,
weights SHA-256 `ced967c4...`, MIT). The catalog, retrieval, candidate ceiling,
reranking depth, blend weight, state, question policy, document serializer,
output size and evaluator are unchanged. It uses no target, sample, or label
information at runtime, and requires no network access at inference.

The Cycle 3 failure audit motivates it: every released-public and screening miss
already enters the 120-candidate pool (`not_retrieved` is 0, recall@120 is 1.000),
and increasing reranking depth did not convert that pool coverage into rank. The
hypothesis is that the reranker's ranking quality, not its input depth, is the
binding constraint. Its additional compute cost is a liability to be disclosed,
not a reason to relax measurement.

## Pre-registered decision boundary

Evaluate once on the Cycle 3 screening split against the recorded 30-depth C0
screening control. The candidate earns confirmation only with all of:

- at least `+0.010` screening TechnicalScore against C0;
- no more than `-0.010` screening Hit@10 against C0;
- valid responses on every turn, with no fallback, agent error, or source drift;
- warm p95 latency at most **3.000 seconds per turn on the measuring machine**.

The latency ceiling is registered before any candidate latency was observed. It is
an absolute per-machine budget rather than a multiple of the control, because the
organizer states that a timeout may be scored as a miss, and a miss is the most
expensive single outcome in the score. The matched control on the same machine is
p95 1.4409 s, so this ceiling is approximately twice the control.

## Released-public run

The released public 200 is already fully consumed and cannot satisfy any gate. A
single public run is descriptive only: it may reject an obviously worse candidate
before any fresh split is opened, but it cannot promote one, repair a failed
screening gate, or be relabelled as independent evidence.

## Scope limits

No further reranker variants, blend weights, or depths follow from this arm
regardless of outcome. Confirmation and validation packs remain unopened.

## Enforced turn budget

`turn_budget_seconds` is an optional per-turn wall-clock budget, disabled by
default (`0.0`), so every previously recorded result reproduces unchanged. When
set, the agent measures observed seconds per reranked candidate and shrinks the
reranking prefix to what the remaining budget affords, skipping reranking and
recording a `latency_budget` fallback rather than overrunning. It exists because
the organizer states a timeout may be scored as a miss, and a miss is the most
expensive single outcome in the score.

The measurement arm keeps the budget disabled, so that the swap remains a
one-field change and its true cost is observed rather than masked. If the arm
fails only the latency ceiling, a budgeted variant may be registered separately;
a budgeted run is not a substitute for the failed unbudgeted one.

## Recorded deviation

The released-public candidate run started before the turn-budget work landed, so
its manifest reports `source_changed_during_run: true`. Its measured behaviour is
unaffected, because the evaluated process imported the pre-change modules and the
budget is inert at `0.0`, but the run does not meet the no-drift condition and is
reported as an indication only. Any run used for a gate decision must be executed
on an unmodified tree.
