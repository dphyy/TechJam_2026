# Neural reranker weight tuning

Run on 30 August 2026. Decision: **keep neural weight `0.75`**.

## Leakage boundary

A local protocol was written before candidate outcomes were observed. It fixed the candidate grid, selection rule, and promotion gate:

- Tuning data: Cycle 5 screening, 160 sessions, SHA-256 `23af9ff1b23889c42c9c53c97e7eab087dededdf3c8b18d0aa36b7414d9265ad`.
- Candidate weights: `0.60`, `0.70`, `0.80`, and `0.90` against the `0.75` control.
- Every other configuration field, source file, dataset, model, and session order remained identical.
- A candidate required at least `+0.005` TechnicalScore, no HitRate@10 or MRR loss, zero fallback/error/source drift, and acceptable runtime health.
- The 80-session confirmation set was reserved for one selected candidate only.
- The 80-session validation set remained unopened and was reserved for one final check after confirmation.

Because no screening candidate passed, confirmation and validation were not evaluated. This prevents the held-out outcomes from influencing parameter selection.

## Screening results

| Neural weight | HitRate@10 | MRR | MTTC | TechnicalScore | Score delta |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.943750 | 0.606391 | 3.443750 | 0.804917 | -0.001349 |
| 0.70 | 0.943750 | 0.606017 | 3.418750 | 0.805305 | -0.000961 |
| **0.75 control** | **0.943750** | **0.608805** | **3.412500** | **0.806266** | — |
| 0.80 | 0.943750 | 0.620404 | 3.425000 | 0.809496 | +0.003230 |
| 0.90 | 0.943750 | 0.621401 | 3.400000 | 0.810295 | +0.004029 |

Paired 10,000-resample bootstrap intervals for TechnicalScore delta:

| Neural weight | Paired delta | 95% interval |
|---:|---:|---:|
| 0.60 | -0.001349 | [-0.010455, 0.007914] |
| 0.70 | -0.000961 | [-0.008860, 0.006708] |
| 0.80 | +0.003230 | [-0.001752, 0.008709] |
| 0.90 | +0.004029 | [-0.004484, 0.012711] |

All arms tied HitRate@10, recorded zero fallbacks and agent errors, used identical non-weight configuration and source hashes, and reported no source drift. Serial p95 measurements varied despite identical neural work, so they should not be interpreted as a causal effect of the blend weight; no candidate reached the quality threshold regardless.

The higher-weight direction improved screening MRR, but the best observed gain was below the preregistered practical threshold and its uncertainty interval crossed zero. Adding a new post-outcome value such as `0.85` or `0.95` would turn this bounded grid into adaptive screening. Test that direction only in a newly registered experiment with new development evidence.

Machine-readable aggregates are in [neural-weight-tuning-results.json](neural-weight-tuning-results.json).
