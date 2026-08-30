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
