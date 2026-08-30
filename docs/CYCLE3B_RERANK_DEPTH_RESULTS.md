# Cycle 3B reranker-depth result

Recorded 27 August 2026 under [the registered depth protocol](CYCLE3B_RERANK_DEPTH_PROTOCOL.md). D60 changes only `rerank_limit`, from 30 to 60. It is not selected: it missed the fresh-screening promotion gate, so the confirmation and validation splits remain unopened.

| Dataset and arm | TechnicalScore | Hit@10 | MRR | MTTC | Prefix recall | Warm p95 | Prompt tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh screening C0 (30) | 0.811786 | 0.925000 | 0.635119 | 3.062500 | 0.937500 | 0.337840 s | 1,723,015 |
| Fresh screening D60 | 0.813490 | 0.925000 | 0.640801 | 3.062500 | 0.968750 | 0.607575 s | 3,608,146 |
| Released public C0 (recorded) | 0.786724 | 0.895000 | 0.613746 | 3.245000 | n/a | n/a | n/a |
| Released public D60, descriptive | 0.797064 | 0.910000 | 0.616212 | 3.140000 | 0.970000 | 0.614152 s | 4,940,799 |

Fresh-screen paired D60 delta is `+0.001705` (exploratory 95% bootstrap interval `[-0.016490, 0.020275]`, 10,000 resamples, seed `20260826`), below the registered `+0.010` promotion gate. The public descriptive delta is `+0.010340` (`[-0.002164, 0.025127]`) with +0.015 Hit@10. Both D60 manifests report no source drift, agent errors, or fallbacks. Screening p95 is 1.80 times C0; public D60 p95 is a local feasibility observation, not an organizer limit.

The public result is development evidence only. It is 0.002936 below 0.80, does not forecast organizer-private performance, and cannot override the failed fresh-screening gate. No confirmation or validation result was accessed.
