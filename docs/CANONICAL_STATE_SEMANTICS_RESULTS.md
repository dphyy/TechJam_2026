> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Canonical state semantics results

## Decision

Phase 8 passes its phase-specific screening gate as a correctness-only
foundation. It remains behind `canonical_state_semantics`; the selected release
is unchanged and no score-improvement claim is made.

## Failure classification

The first authored suite contained four reported state differences. Three were
genuine semantic problems:

- a correction marker did not carry across punctuation, so a later changed
  material could coexist with the old material;
- “no longer have a color preference” was not recognized as a color-scoped
  neutral statement;
- an alternative’s turn/clause group ID and discourse word `suitable` leaked
  surface order into semantic state.

The remaining case compared “I need …” with “would be great”. The former is a
hard requirement and the latter is soft, so equality of hard constraints was an
invalid property. The v2 fixture retains equal modality while varying clause
order, punctuation, case, hyphenation, and inflection.

## Candidate behavior

The candidate:

- derives stable alternative identities from attribute and sorted values;
- sorts canonical active facts before query and retrieval-plan construction;
- carries an explicit correction across punctuation only for the attributes
  actually restated;
- scopes “no longer have … preference” to its named attribute;
- preserves unrelated facts, polarity, hardness, component scope, and source
  provenance;
- keeps raw source text available for diagnostics and neural ranking.

The frozen eight-family v2 suite passed every state, plan, membership, override,
no-change, missing-metadata, and legality property. Minimum Top-120 Jaccard,
Top-10 overlap, and rank correlation were each `1.0`.

## Fresh screening

| Metric | Selected control | Canonical candidate | Delta |
|---|---:|---:|---:|
| HitRate@10 | 0.981250 | 0.981250 | 0.000000 |
| MRR | 0.661453 | 0.654856 | -0.006597 |
| MTTC | 2.218750 | 2.225000 | +0.006250 |
| TechnicalScore | 0.864686 | 0.862582 | -0.002104 |
| p95 | 0.411602s | 0.485555s | +0.073953s |
| Prompt tokens | 1,539,346 | 1,512,580 | -26,766 |
| Fallback turns | 0 | 0 | 0 |

There were zero paired gained or lost hits. Both arms had the same three misses:
two admission failures and one retrieval failure. All critical scenario
HitRates were unchanged. The score decrease is within the explicit Phase 8
`0.003` allowance, but MRR and the sequential p95 measurement are weaker; this
is why the candidate is recorded as correctness-only rather than promoted as a
quality or efficiency win.

Confirmation remains sealed. Later candidates that depend on canonical state
must compare against this candidate on open evidence and must still earn their
own independent gates.
