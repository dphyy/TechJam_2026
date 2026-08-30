# Margin-aware fusion screening result

Decision: **Rejected at screening; do not promote.**

The comparison followed `docs/MARGIN_FUSION_PROTOCOL.md` without changing its threshold or gate after outcomes were observed. Both arms used the same source tree, catalog, and 160-session Cycle 5 popularity-matched screening pack.

| Metric | Selected control | Margin-fusion candidate | Candidate delta |
|---|---:|---:|---:|
| Hit rate @ 10 | 0.943750 | 0.943750 | 0.000000 |
| MRR | 0.608805 | 0.603259 | -0.005546 |
| MTTC | 3.412500 | 3.481250 | +0.068750 |
| Technical score | 0.806266 | 0.803228 | -0.003038 |
| p95 turn latency | 0.696662 s | 0.700289 s | +0.52% |
| Fallback turns | 0 | 0 | 0 |
| Agent error turns | 0 | 0 | 0 |

The candidate tied hit rate and met the runtime-health constraints, but it failed all three quality conditions: it did not gain at least 0.005 technical score, and both MRR and technical score declined. Confirmation and validation were therefore not run.

The complete immutable receipts are stored under:

- `runs/cycle5-margin-screen-control-20260830/`
- `runs/cycle5-margin-screen-candidate-20260830/`

The implementation remains behind `neural_margin_fusion: false` so the experiment is reproducible without changing the selected release.
