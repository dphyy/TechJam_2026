> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Catalog-Derived Vocabulary Protocol

## Registered hypothesis and boundary

A versioned alias table derived only from the supplied catalog can improve slot
recall for unseen category wording without weakening deterministic negation,
correction, component scope, no-preference, or unknown-metadata behavior.

The candidate is opt-in through `catalog_vocabulary`. Static explicit parsing
always owns overlapping spans. The catalog matcher examines only spans left
unowned by that parser. Its suggestions are additive, soft evidence and cannot
retire a conflicting explicit ledger fact; explicit negative wording remains a
hard exclusion. Missing fields remain unknown.

The candidate source/config and the cases below are frozen before opening the
fresh end-to-end comparison.

## Artifact construction

- Version: `catalog-vocabulary-20260830-v1`
- Minimum support: 5 catalog rows
- Minimum confidence: 0.80 for a unique alias → canonical mapping
- Catalog SHA-256:
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`
- Model SHA-256:
  `bd770613ed9f85016721bb9be4edce7591fca04eed962f87faa254f4c7d6e43e`
- Frozen word-case SHA-256:
  `739c8880548be736cba2b5d557939a24da687731db22b6cb953bff73c723b815`

The artifact contains 1,399 unambiguous aliases: 1,148 category, 102 material,
83 style, and 66 color aliases. It normalizes case, punctuation, whitespace,
hyphenation, and conservative final-token singular/plural variants. Cross-
attribute alias collisions are removed rather than guessed.

Canonical values come from category paths and reliable structured detail keys;
title occurrences may confirm category-path evidence. A separate catalog-path
taxonomy labels sufficiently supported category values as object, accessory, or
component. No public failure title, evaluator target, dialogue outcome, or model
ranking contributes to the artifact.

The model is bound to exact catalog bytes. A missing, malformed, ambiguous, or
catalog-mismatched artifact records `catalog_vocabulary` fallback and preserves
selected behavior.

## Frozen word evaluation

Forty-eight category canonical families are hash-selected from aliases absent
from the static parser. Cases with a known static alias anywhere inside the
span are excluded, preventing the existing parser from receiving partial credit.
Four independently named template families phrase the requests. Authors did not
inspect Mercury rankings or evaluator outcomes.

| Arm | Slot precision | Slot recall | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Selected static parser | 1.000000 | 0.000000 | 0 | 0 | 48 |
| Catalog vocabulary | 1.000000 | 1.000000 | 48 | 0 | 0 |

This exact-lookup suite proves coverage of the registered catalog aliases, not
general natural-language understanding. The full deterministic and metamorphic
suites remain the precision and interaction gate.

## Promotion gates

- Full unit, private-like, metamorphic, legality, and failure tests stay clean.
- No precision regression on negation, correction, component scope, or
  no-preference tests.
- No new unsupported hard positive exclusion.
- Fresh source-matched HitRate@10 and TechnicalScore are non-negative.
- No unexpected fallback and no material p95 or RSS regression.
- `configs/selected.json` remains unchanged until the fresh result passes.

## Frozen end-to-end result

Confirmation opened once at source commit `afa2df7` against dataset SHA-256
`74c17858370e6e7bb30d15d4d6cf28289d86d52d9a72a21466189f3112f972a3`.
There were no startup or turn fallbacks.

| Arm | HitRate@10 | MRR | MTTC | TechnicalScore | p95 |
|---|---:|---:|---:|---:|---:|
| Selected | 0.937500 | 0.591310 | 3.237500 | 0.801393 | 0.347s |
| Catalog vocabulary | 0.950000 | 0.605045 | 3.050000 | 0.815514 | 0.381s |

The candidate improved every aggregate quality metric and its 9.8% p95 increase
was much smaller than the rejected all-pool scorer. Boundary and override
HitRate were unchanged; browsing improved from `0.857143` to `0.928571`.
However, buying HitRate fell from `0.967742` to `0.935484`—one lost session out
of 31. The predeclared critical-slice rule does not permit that loss to be hidden
by the aggregate gain.

Phase 5 is therefore not promoted. The artifact and candidate config remain
available for future evaluation on genuinely new evidence, but confirmation is
consumed and cannot be used to tune aliases or thresholds. Final remains sealed
and `configs/selected.json` is unchanged.
