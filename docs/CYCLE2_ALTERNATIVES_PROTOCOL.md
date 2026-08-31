> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Bounded alternatives experiment

Registered after the role/scope kill test and the disclosed development characterization, before alternatives implementation or its model comparisons. The prior H1 gate is not relaxed or reused. All previously created validation packs remain locked and unchanged.

## Hypothesis and controls

H4: explicit same-attribute alternatives can preserve shopper intent and prevent false observed-contradiction penalties better than a parser-only repair, without a material regression on ordinary target recovery.

Three configurations:

- Frozen: unchanged behavior from `99095211a1732eb750610bd59610f215dee3136f`.
- Parse-only: preserve explicitly enumerated alternatives when ordinary no-preference wording would swallow them; keep the original independent constraint guard.
- Grouped: the same parsing repair plus an explicit OR group for positive same-attribute values connected by an unambiguous “or”. Require every live alternative to have observed contradictory evidence before the OR group can be declared contradicted. Independent exclusions and conjunctions remain separate.

No new model, model calls, candidate budget, token budget, question policy or inference-visible data. Keep all previous local engineering limits and old-public regression floors. Do not add product IDs, category-specific label fixes or evaluation-template recognition.

## Bounded language and evidence

Support enumerated known facet values within one clause. Ambiguous mixed conjunctions, cross-attribute Boolean expressions, nested expressions and product/component scope are not claimed. Negative “neither/nor” requirements remain conjunctions of exclusions. “Any color works” must still clear color, while an explicit list such as “either black or brown works” must preserve the listed alternatives. A later correction replaces the relevant option set; an explicit rejection removes that option without deleting unrelated requirements.

Group transitions are atomic. A plain restatement of one option preserves its membership; an explicit correction choosing that option retires the rest of the group. Overlapping new option lists without an explicit replacement are unsupported: preserve the existing choice set for that attribute, report the unsupported interpretation, and continue processing unrelated attributes. Explicit replacement retires the old group before installing the new one.

For grouped constraints, support for any live option satisfies the group. A mixture containing an unknown option is not a proven contradiction. Only contradiction of every live option allows a group penalty. All candidates remain in the diagnostic pool; the guard changes rank, not catalog membership. Inspectable diagnostics expose group membership and the existing source turn/text through session state; do not generate unsourced product claims.

## Development gate

Use the eight already visible capability cases plus a declared developer regression suite. The latter covers the following scenario classes before reading outcomes: explicit lists versus unrestricted preference, hard alternatives with one supported/one contradicted option, all alternatives contradicted, one unknown alternative, independent negative constraints, ordinary conjunctions, corrections, option rejection, neutralization, cache invalidation and unsupported mixed expressions. Report this suite as developer-authored tests, never independent validation.

The grouped variant must pass every deterministic OR truth-table and lifecycle invariant, improve at least one false-contradiction case over parse-only, and introduce no new invalid-ID, false-contradiction, stale-state or unrelated-preference-loss failure in those checks. The existing eight-case suite must not lose any previously passed assertion. Pairwise ranking assertions with a missing comparator are failures, not silently passed; report this retrieval limitation separately from ranking correctness.

Run all 200 old public development sessions and 32 fresh-target development sessions with fixed configurations and actual model assets. Select the grouped variant only if it meets the original public regression floors (TechnicalScore loss no greater than 0.01, HitRate loss no greater than 0.02) and local resource limits. Claim a broad score improvement only under the original minimum gain and uncertainty rules. Otherwise keep the minimal verified repair or frozen baseline and report the narrower result.

## Locked validation

Before opening validation content, freeze source, tests, adapter and these three configurations. Evaluate all three once on the same 32 new-target validation sessions and 24 author-separated capability cases. Record consumption before model inference. A failed run remains recorded; recovery may only address infrastructure with unchanged source/configuration. Any correctness change after outcomes consumes the set for that changed implementation.

Locked acceptance is fixed before opening outcomes: the proposed release must lose no more than 0.01 TechnicalScore and 0.02 HitRate against frozen behavior on the 32 target-recovery sessions, and introduce no newly failing assertion relative to frozen behavior on the 24 capability cases. It must also retain the registered local resource limits. A grouped candidate that fails any gate is not promoted. Parser-only is the fallback only if it independently meets these same locked gates and its development correctness tests; otherwise retain frozen Mercury. These small-sample floors are conservative release decisions, not evidence of statistical equivalence.

The existing capability lock has only three cases per group. It may contain little or no coverage of the new hard-OR mechanism. Report that coverage explicitly; an all-pass result on unrelated cases does not validate Boolean correctness. Do not create additional supposedly independent tests after inspecting their outcomes. Preserve the distinction between developer regressions, author-separated synthetic cases, and the unchanged official simulator.

## Demo and stop rule

Three label-free real-catalog demonstration probes are fixed before their inference results, in this order: (1) “I need a bag. It must be canvas or leather.” -> “Actually, canvas only. Keep the bag requirement.”; (2) “I need a shirt. It must be cotton or linen.” -> “Actually, no linen.”; (3) “I need a jacket. It must be waterproof or insulated.” -> “Actually, waterproof only.” Run all three under every control, keep all responses, and select the first with a source-verifiable grouped-constraint intervention. These are developer-authored demonstrations over the real catalog, not independent shopper tests or exact-target benchmarks.

Show an actual backend exchange that preserves two explicit alternatives, rejects one and changes the ranking/state without reviving stale preferences. If no real-catalog development case exhibits a source-verifiable grouped-constraint intervention over parse-only, use a clearly labeled invented mini-catalog demonstration and say that a real-catalog benefit was not observed. Show an unknown-evidence example too.

Do not pursue another new mechanism during this cycle if the fallback fails. Deliver the verified baseline, the reproduced failure and the honest experiment result. Preserve the blank README and private-only publication boundary.
