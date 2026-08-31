> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Cycle 4 results: reranker swap rejected, slate paging passes screening

Recorded 28 August 2026 under the registered
[reranker-swap](CYCLE4_RERANKER_SWAP_PROTOCOL.md) and
[slate-paging](CYCLE4_SLATE_PAGING_PROTOCOL.md) protocols. Measuring machine:
Windows 11, CPython 3.11.4, four CPU threads. Catalog SHA-256 `da979b05...`,
screening pack SHA-256 verified against its lock.

## Measurement integrity

The selected configuration reproduces both recorded references exactly on a
different operating system, Python version and CPU:

| Dataset | Measured here | Recorded reference |
|---|---:|---:|
| Released public 200 | 0.786724 | 0.786724 |
| Cycle 3 screening 160 | 0.811786 | 0.811786 |

Hit@10 also matches exactly on both (`0.895000`, `0.925000`). The comparison
below is therefore against a control measured on the same machine, and the
pipeline is confirmed deterministic across platforms.

## Rejected: `bge_reranker_base`

One field changed, `reranker_model`, from `cross-encoder/ms-marco-MiniLM-L6-v2`
to `BAAI/bge-reranker-base`. Released public 200, descriptive only.

| Metric | MiniLM control | bge-reranker-base | Delta |
|---|---:|---:|---:|
| TechnicalScore | 0.786724 | 0.789830 | +0.003106 |
| Hit@10 | 0.895000 | 0.910000 | +0.015000 |
| MRR | 0.613746 | 0.591768 | **-0.021978** |
| MTTC | 3.245000 | 3.135000 | -0.110000 |
| Warm p95 | 1.4409 s | **9.5004 s** | 6.6x |

Rejected on two registered gates. The score gain is `+0.003` against a `+0.010`
threshold, measured on the consumed public set that has historically flattered
candidates: D60 and D120 scored `+0.010` and `+0.020` here and then failed fresh
screening at `+0.002` and `-0.003`. The warm p95 of `9.5004 s` is more than three
times the registered `3.000 s` ceiling.

The larger cross-encoder raised Hit@10 while *lowering* MRR: it pulls additional
targets into the slate but reorders items the smaller model already placed at
rank one or two. Because MRR carries `0.30` weight, the ordering loss cancels most
of the coverage gain. The registered hypothesis, that reranker quality was the
binding constraint, is not supported. No screening run was spent on it.

## Passed screening: `slate_paging_first_turn = 5`

One field changed, `slate_paging_first_turn`, from `0` to `5`. Evaluated once on
the Cycle 3 screening split against the C0 control reproduced above.

| Metric | C0 control | Paging | Delta |
|---|---:|---:|---:|
| TechnicalScore | 0.811786 | 0.836788 | **+0.025002** |
| Hit@10 | 0.925000 | 0.962500 | +0.037500 |
| MRR | 0.635119 | 0.647210 | +0.012091 |
| MTTC | 3.062500 | 2.931250 | -0.131250 |
| Warm p95 | 2.3799 s | 2.4316 s | +2.2% |

Paired 160-session bootstrap, 10,000 resamples, seed `20260826`:
delta `+0.025002`, 95% CI `[0.007830, 0.046629]`.

Registered gate, all four conditions met:

| Condition | Required | Observed | |
|---|---|---|---|
| Screening TechnicalScore | >= +0.010 | +0.025002 | pass |
| Screening Hit@10 | >= -0.010 | +0.037500 | pass |
| Fallbacks, agent errors, source drift | none | 0, 0, false | pass |
| Warm p95 | <= control +10% | +2.2% | pass |

Retrieval failures are unchanged at 2 in both arms; ranking or policy failures
fall from 10 to 4. Paging repairs six of the ten sessions where the wanted
product was retrieved and ranked but never displayed.

The CI lower bound of `+0.007830` sits just below the `+0.010` practical
threshold. The registered gate is stated on the point estimate and is met; a
stricter reading requiring the entire interval to clear `+0.010` would not be.
The interval does exclude zero.

## Released-public implementation check

The protocol pre-registered an exact counterfactual prediction before the code
existed. The implementation reproduced it to six decimal places:

| Metric | Predicted | Measured |
|---|---:|---:|
| TechnicalScore | 0.839176 | 0.839176 |
| Hit@10 | 0.970000 | 0.970000 |
| MRR | 0.645900 | 0.645919 |
| MTTC | 2.980000 | 2.980000 |

This validates the replay method and the implementation. It remains descriptive.

## Limitations

- Screening has now absorbed one further comparison. Confirmation and validation
  remain unopened.
- Screening targets are drawn by hash from the whole frozen catalog. The
  organizer's public and private sessions are drawn from 1,406 candidate targets
  that survived a 5-core leave-last-out join, so screening over-represents
  obscure listings. Both arms face identical sessions, so the comparison holds,
  but the absolute level is not an estimate of private performance.
- Local scores are measured against intent cards reconstructed by the
  participant evaluator, not the organizer's own intent cards.
- The bge run carries `source_changed_during_run: true` and is reported as an
  indication only; every other run in this cycle is drift-free.
