# Early paging with override reset result

Decision: **Promoted.**

The comparison followed `docs/EARLY_PAGING_OVERRIDE_RESET_PROTOCOL.md`. Both
arms used identical source, catalog and public-development data hashes, and
neither source tree changed during its run.

| Metric | Turn-5 control | Early paging + override reset | Candidate delta |
|---|---:|---:|---:|
| HitRate@10 | 0.970000 | 0.970000 | 0.000000 |
| MRR | 0.645919 | 0.641633 | -0.004286 |
| MTTC | 2.980000 | 2.905000 | -0.075000 |
| Efficiency | 0.802000 | 0.809500 | +0.007500 |
| TechnicalScore | 0.839176 | 0.839390 | +0.000214 |
| p95 turn latency | 0.350255 s | 0.583208 s | +66.51% |
| Prompt tokens | 2,375,969 | 2,372,103 | -3,866 |
| Evaluated turns | 590 | 575 | -15 |
| Fallback / agent-error turns | 0 / 0 | 0 / 0 | 0 / 0 |

The owner's registered promotion rule was TechnicalScore greater than or equal
to control. The candidate passes narrowly. HitRate stays at 194/200 and earlier
ordinary-session recovery improves MTTC enough to offset lower overall MRR.
The MRR and paired-run p95 regressions are explicit trade-offs; this result is
not evidence that every component improved.

Intent-override HitRate, MRR and MTTC match control exactly at `0.900000`,
`0.748611` and `4.533333`. The three targets lost by unguarded early paging in
`public_0071`, `public_0103` and `public_0183` reset to page 1 on their override
turns and are recovered. The reset diagnostic is `slate_page_reset:
"intent_override"`.

Complete receipts are retained under:

- `runs/override-reset-control-20260830/`
- `runs/override-reset-candidate-20260830/`

The selected release now uses `slate_paging_first_turn: 1` together with
`slate_reset_on_override: true`. This is consumed public development evidence,
not a private-test forecast.
