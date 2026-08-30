# Catalog vocabulary v2 results

## Decision

Phase 9 is rejected at screening. The dual-lane design fixes v1’s persistence
and precision risks, but it does not improve end-to-end TechnicalScore and it
breaches the neural-token cap. Confirmation remains sealed and the candidate
will not be combined with later arms.

## Frozen artifact and behavior

The catalog-hash-bound v2 artifact contains 1,414 aliases, of which 980 are
eligible for the state lane only when an explicit local attribute or product
request cue is present. Every record includes support, confidence, ambiguity
margin, state eligibility, taxonomy role, and extraction method.

The runtime lanes are:

1. A high-confidence, sufficiently unambiguous, explicitly cued alias may add a
   soft catalog-sourced fact. Exact static-parser spans retain ownership.
2. Other exact unambiguous aliases are bounded current-turn BM25 expansions.
   They are not preferences, direct rank boosts, hard constraints, or session
   memory.

Negated aliases are suppressed from both lanes. Accessory and component roles
cannot become whole-product category facts. Retrieval expansions disappear on
the next turn and cache keys include their exact query, so correction,
no-preference, and override transitions cannot retain stale expansion state.

The 144-case frozen suite contains 48 state positives, 48 retrieval-only
positives, and 48 negated or ordinary-context adversarial cases. State
precision/recall and combined dual-lane coverage precision/recall were each
`1.0`. This exceeds the `0.99` state-precision requirement and preserves v1’s
unseen-alias coverage in the larger suite.

## Screening result

The correct source-matched control is the Phase 8 canonical-state candidate.

| Metric | Canonical control | Vocabulary v2 | Delta |
|---|---:|---:|---:|
| HitRate@10 | 0.981250 | 0.981250 | 0.000000 |
| MRR | 0.654856 | 0.654360 | -0.000496 |
| MTTC | 2.225000 | 2.262500 | +0.037500 |
| TechnicalScore | 0.862582 | 0.861683 | -0.000899 |
| p95 | 0.485555s | 0.395458s | -18.55% |
| Prompt tokens | 1,512,580 | 1,710,586 | +13.09% |
| Peak RSS | 1,179,467,776 | 1,115,209,728 | -5.45% |
| Fallback turns | 0 | 0 | 0 |

All overall and scenario HitRates were unchanged, and the three misses remained
at the same retrieval/admission stages. However, Phase 9 requires a positive
TechnicalScore delta and no more than 3% additional serialized neural tokens.
It fails both requirements. The lower p95 cannot rescue the failed quality and
token gates.
