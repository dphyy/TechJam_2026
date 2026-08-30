# Cycle 3C full-pool reranking result

Recorded 27 August 2026 under [the registered full-pool protocol](CYCLE3C_FULL_POOL_PROTOCOL.md). D120 changes only `rerank_limit`, from 30 to 120, so the local reranker scores every candidate retained by the selected sparse pipeline.

| Dataset and arm | TechnicalScore | Hit@10 | MRR | MTTC | Prefix recall | Warm p95 | Prompt tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh screening C0 (30) | 0.811786 | 0.925000 | 0.635119 | 3.062500 | 0.937500 | 0.337840 s | 1,723,015 |
| Fresh screening D120 | 0.809141 | 0.925000 | 0.624638 | 3.037500 | 0.968750 | 1.339450 s | 7,453,902 |
| Released public C0 (recorded) | 0.786724 | 0.895000 | 0.613746 | 3.245000 | n/a | n/a | n/a |
| Released public D120, descriptive | **0.807170** | **0.925000** | **0.615567** | **3.000000** | 1.000000 | 1.208881 s | 10,010,567 |

The fresh-screen paired D120 delta is `-0.002644` (exploratory 95% bootstrap interval `[-0.021330, 0.016515]`, 10,000 resamples, seed `20260826`), failing the `+0.010` promotion gate. Its screening p95 is 3.97 times C0, also beyond the registered two-times cap. D120 is therefore disqualified from confirmation, validation, and selection.

The pre-registered one-time public descriptive result is `+0.020446` TechnicalScore against the recorded C0, with +0.030000 Hit@10 and exploratory paired 95% bootstrap interval `[0.003157, 0.040038]` (10,000 resamples, seed `20260826`). It is 0.007170 above the immediate 0.80 public-score target. The completed run had no source drift, fallback, agent error, or retrieval miss; it used 10,010,567 local model prompt tokens, p95 1.208881 seconds, peak RSS 1,110,867,968 bytes, and 475.418 seconds evaluator time.

This is the highest released-public score measured in this campaign, not a claim about organizer-private performance or a release recommendation. The selected production configuration stays at 30 reranked candidates because every new mechanism and full-pool D120 failed its fresh-screening decision boundary. The confirmation and validation packs remain unopened.
