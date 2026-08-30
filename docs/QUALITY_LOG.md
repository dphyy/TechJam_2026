# Quality log

## Initial integration

- Test-first failures observed before each new module and material regression fix.
- Full unit/integration run reached 133 passing tests before iteration two.
- Integration review inspected state and policy code and reran focused tests.
- Default Ruff checks passed across runtime, experiment, baseline and test code (`ruff 0.9.6`).
- No type-checker configuration exists in the starter. Static type checking has not been claimed; compilation and runtime tests are separate checks.

## Behavior-preserving simplification pass

Three read-only review scopes covered reuse, code quality and efficiency. Six findings were applied:

1. Reuse the streaming file-hash utility in split preparation; retain `hashlib` for deterministic sample partitioning.
2. Remove the unused catalog position map.
3. Remove an unused test import.
4. Clarify that source text is retained in session memory, never written to disk by the state module.
5. Use bounded `heapq.nsmallest` for contrast neighbors, preserving the deterministic similarity/ID key.
6. Group contrast evidence by attribute once, preserving score accumulation and ordering.

Applied counts: reuse 1, quality 3, efficiency 2. One efficiency suggestion, maintaining a second mutable active-preference cache, was skipped because it added synchronization complexity to a small turn-bounded ledger. Validation/checksum safeguards and official entry wrappers were retained.

Focused contrast, retrieval and split tests passed after these changes. The dedicated review below subsequently covered correctness, security, API compatibility and reliability.

## Additional regression found during integration

The neural reranker normalized its top prefix but left the unreranked tail on the prior evidence-score scale. This did not change direct slate ordering, but could mislead question lookahead. A failing monotone-score regression was added, then tail scores were placed below the prefix on a consistent scale. The focused neural tests pass.

## Formal review and corrective pass

Seven independent local review scopes covered correctness, tests, maintainability, security, performance, API contracts and reliability. The review receipt completed with `Ready with fixes`; a separate validation pass confirmed four actionable findings. All four were addressed before the third development iteration:

1. Failure diagnostics now join unchanged official per-session hit outcomes by sample ID. Raw recommendation membership before an override is not counted as an official hit. This fixes diagnostic attribution, not the official metric calculation.
2. Product evidence now recognizes direct negation and material qualifiers at matching spans. Explicit support, contradiction, unknown and mixed evidence remain distinct. An always-on, conservative constraint guard demotes confirmed contradictions without filtering unknown metadata. Positive evidence boosts remain independently ablatable.
3. Same-message corrections are applied in clause order, including dependent quantity invalidation. A withdrawn fact is not accidentally reactivated by a later clause.
4. The current message is clipped once before all normal and fallback paths. Optional malformed assets/rankings are handled consistently; cached fallback provenance is preserved.

Regression coverage includes synthetic catalog constraints, unknown prices, conflicting source evidence, repeated guard application, no-information caching, budget changes, local entrypoint loading and injected optional-module failures. All 211 unit/integration tests and Ruff passed in the fresh environment before iteration 3. No repository type-checker is configured; no static type-check result is claimed.

An external cross-model adversarial review was attempted twice but failed before review because its CLI rejected `--safe-mode`. No external model result or independent cross-model assurance is claimed. The completed local review and validation are the usable evidence; the degraded external lens remains a limitation.

## Fresh environment and offline execution

Created `.venv-repro` with CPython 3.12.12, installed the then-current split requirements files, and ran `pip check` successfully. Current setup has since been consolidated into the single `requirements.txt`; historical aggregate hashes still name the old files to preserve the original evidence record.

Both pinned models loaded and performed real inference in this fresh environment under macOS `sandbox-exec` with `(deny network*)`, with `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` removed. A separate socket probe confirmed OS-level network denial. BGE produced a `(1, 384)` embedding; the reranker ranked two invented products and reported 21 input tokens. This is an actual model smoke test, not yet the final full official-harness reproduction.

Dense matrix scoring uses a bounded finite-checked `einsum` operation after the local BLAS path emitted spurious floating-point warnings on finite normalized vectors. Its deterministic score/tie behavior is tested. Raised timeouts are handled; an indefinitely stalled native model kernel still requires an external process timeout.

## Final development failure audit

The iteration-3 neural control had 33 official misses. An evaluator-side audit found generic linguistic edge cases: a lexical no-show compound misread as negation, an avoidance instruction misread as positive product evidence, generic metadiscourse inheriting color ownership, and a broad unanswered `other` question clearing earlier explicit details. The final bounded correctness pass addressed these with invented-message/catalog regressions before candidate confirmation; it did not use reserved outcomes or add target-specific rules. Root verification passed all 223 tests, Ruff and `git diff --check` before iteration 4.

One observed requested-material/catalog conflict was genuine: a hard leather request paired with a catalog item described as faux leather. The conservative guard is retained even when respecting visible requirements can hurt exact-target scoring. Matching runtime-visible metadata found no exact catalog peer for the 33 failed targets in that audited control; catalog ambiguity therefore was not asserted as the cause of those misses.

## Frozen release reproduction

- Froze both candidate configs and all runtime/experiment/test/demo sources before reserved evaluation; each finalist ran once. Selected the lower-cost 30-prefix system without changing its source or parameters.
- Actual official CLI in `.venv-repro`, with OS networking denied and named provider credentials removed: 179/200 hits, TechnicalScore 0.786724, exit 0. All 200 session records exactly match the frozen development/reserved union. Whole-command wall time 148.32s; maximum RSS 1,241,497,600 bytes.
- Reproduced the untouched starter again under the same network-denial wrapper: 25/200 hits, score 0.106710, exactly matching the reference.
- Generated a real three-turn terminal replay with networking denied. Inspected its queries, exclusions and returned titles; documented the remaining soft-preference/accessory ranking limitations instead of claiming perfect fit. The `.cast` is a narration-paced terminal recording, not an MP4 or public upload.
- A separate documentation audit corrected the cold-start timing definition and optional-question wording. No runtime changes followed the source freeze.
- Full-catalog automatic fallback was also exercised with the selected neural configuration pointed at an empty temporary asset directory, under OS network denial. It recorded the missing reranker, returned ten unique valid catalog IDs, and reported zero model tokens. A loopback socket probe independently confirmed `PermissionError` from the OS network policy. This was a real missing-file fallback, not a mocked inference result.
- Aggregate evidence is saved in `experiment-summary.json`: 32 completed run manifests/results, eight paired comparisons, final source freeze, exact asset provenance and the whole-public offline reproduction. Raw session traces remain local and ignored by source control.
- Final evaluation audit verified all 32 aggregate rows against raw results, diagnostics and file hashes; all local Markdown links and four demo artifacts resolve. Final tests (223), Ruff, `pip check`, compilation and whitespace checks passed. Official input hashes and frozen source/config still matched; exactly two reserved runs existed. At that evaluation snapshot the Git index was empty, and no commit or publication had occurred. Later source publication is recorded in `WORK_LOG.md`.
- Documentation distinguishes a full-catalog fallback smoke test from exhaustive per-ID testing, and post-install offline execution from an untested fully air-gapped dependency installation.

## Cycle 2: explicit alternatives implementation

The feature remains opt-in while its registered experiment runs. State/configuration tests were written before the new interfaces existed. The ranking tests then reproduced false penalties before the guard was changed. The completed state and ranking slice passed all 288 unit/integration tests, Ruff and whitespace checks at `a05d1ce`; this is not a claim that the later experiment harness or final submission is complete.

A read-only correctness review found two additional state failures: a later “cotton only” retained linen, and an explicit “both cotton and linen” retained the previous OR grouping. Three new test methods reproduced four failing cases before an attribute-scoped correction fix. The suite now checks both OR-to-AND and AND-to-OR changes, stable non-corrective restatement, partial rejection, independent exclusions, unknown evidence and cache invalidation even when query words do not change.

Separate reuse, quality and efficiency passes covered each completed implementation slice. No simplification was applied. One eager-regex optimization was deferred: its material benefit was unmeasured, and moving extraction across early-return paths would need to preserve diagnostic side effects. Validation safeguards and independent Config/SessionState construction checks were retained. No static type-check result is claimed; the repository has no configured type checker.

The matched off-mode run was compared turn by turn against the original frozen-source run on all 200 public sessions. All 628 turns have identical messages, complete responses including usage, queries, revisions, active preferences, cache decisions, candidate routes, ranked IDs and policy diagnostics. Only timings and newly added diagnostic fields were excluded. The complete official result JSON is identical, SHA-256 `eeba444682d430d42bd34e87f58668187f82c21486b4dd791890c6176d85ec5e`. The new run's source hashes remained stable. This establishes observed control parity on that development set, not universal behavioral equivalence.

Validation packs remain unchanged and unopened at this checkpoint. No external source-analysis service was used for these reviews. Subsequent development, locked-validation and replay results must be recorded separately before selecting the release configuration.

## Cycle 2: development and offline checks

At `14d14d1`, the strict capability runner and one-time validation guard pass 25 focused tests; the complete suite passes 313 tests and Ruff. Before implementation, two additional harness regressions reproduced a rejected neutral preference and an assertion-free case incorrectly accepted as valid. The adapter now accepts the ledger's neutral polarity and requires at least one assertion per case. Missing comparison candidates, missing penalty diagnostics and model fallbacks cannot silently become passing capability results.

All three fixed controls completed the unchanged official simulator on both development sets. Public: 179/200 hits and TechnicalScore 0.786724. New targets: 27/32 hits and score 0.751053. Every paired per-session score delta is zero; the registered 10,000-resample bootstrap (seed 20260826) therefore returns [0, 0] on these observed sessions, not a population-equivalence guarantee. The public conversations never activated an OR group, so their unchanged scores establish regression evidence rather than a new-capability benefit. Maximum public response times reached 1.000 seconds for parse-only and 1.423 seconds for grouped, despite p95 values below the one-second local gate.

On the eight authored capability-development cases, frozen passes 12/15 assertions and 6/8 complete cases; parse-only and grouped each pass 14/15 assertions and 7/8 complete cases. Both repair the two lost color preferences in the explicit-choice case, with no lost passes, unverified assertions, invalid responses or fallbacks. All three fail the same ordinary-query comparison because its irrelevant comparator is absent from the retained candidates. This failure is preserved, not reinterpreted as a pass. The extra grouped-constraint benefit is established by developer truth-table regressions, not by this eight-case set.

The grouped configuration also completed a real two-turn, 50,000-product smoke test with OS networking denied and five named provider credentials removed. A loopback bind independently raised `PermissionError`. The healthy run loaded the actual pinned model, used nonzero model tokens and had no fallbacks. Two further full-catalog runs exercised an empty asset directory and an invalid-weights checksum fixture through the real loader. Both recorded the specific neural failure, used zero model tokens, returned ten valid unique catalog IDs each turn and retained the shopper's canvas requirement after rejecting leather. Original model assets were not altered. Total process peak RSS across these serial probes was 848,265,216 bytes. This is six observed response turns, not exhaustive model-failure coverage.

Development comparisons and offline probes ran from immutable source archives while unrelated replay work continued. Exact source/configuration/input hashes and raw responses remain in ignored local artifacts. Locked validation was still unopened at this checkpoint. Private integration into `main` preserves the original commits and does not promote the experimental configuration or authorize public submission.

## Cycle 2: reviewed correction boundary

Seven local review lenses covered the completed source: correctness, testing, maintainability, security, performance, reliability and adversarial failure construction. One independent validation batch confirmed six distinct defects after duplicate reports were merged. A separate read-only check of the applied fixes found all six resolved, with no fix-specific regression identified. No external source-analysis provider was used, and no locked validation content or outcomes were accessed during review.

- Same-message OR-to-only and OR-to-AND corrections now resolve against the live clause-ordered ledger.
- Prefix selections such as “Only linen” retire the old group and invalidate its cached ranking; “not only” remains non-exclusive.
- Rejecting an unsupported overlapping positive list does not discard an independent exclusion from the same attribute batch.
- “Any color works” can clear color while an independent material alternative list is preserved.
- Adding pockets does not erase an unrelated weather-protection choice group.
- Malformed recommendation identifiers and presentation failures retain the original error, actual responses and failed replay manifest.

The state regressions failed before the runtime edit: 11 state-test subcases and nine Agent-test subcases reproduced the reported failures. Three replay regressions then reproduced the unhashable-ID crash, missing manifest and overwritten original error before the presentation fix. Two additional tests cover already-correct capability cleanup and protected validation-output guards; these are coverage additions, not newly fixed runtime defects.

At `8964157`, all 347 tests pass in `.venv-repro`; Ruff, dependency consistency and whitespace checks pass. Target-lock verification is read-only and reports no rewritten outputs or validation-outcome access. A direct comparison also preserves full off-mode state/history behavior on 24 authored traces across ledger/latest/history modes. These checks do not replace the final actual-model comparisons. The corrected source is archived before those development reruns, and the README remains empty.

## Cycle 2: final comparison, selection and evidence audit

All nine final development comparisons and six one-shot validation comparisons completed on the unchanged 60-file source inventory from `8964157`. This count comprises six development target runs, three development capability runs, three validation target runs and three validation capability runs. The exact OFF control preserves all 737 development turns against the original frozen implementation. Every control has the same target results within each set: public 179/200 and score 0.786724; new development 27/32 and 0.751053; different locked validation 31/32 and 0.838032. All paired target differences are zero. No target-recovery conversation activates an OR group.

Capability development is 12/15 assertions for frozen and 14/15 for both repairs. Locked capability validation is 46/51 and 19/24 complete cases for every control, with no new failure or incremental pass. Grouped validation has two active OR turns, including one hard-OR turn; that limited coverage does not distinguish the grouped guard. Two body/component ordering failures and three missing-comparator failures are retained. No source fix followed those outcomes.

The original freeze verifier passed after all six completed validation jobs and immediately before configuration promotion. `2de717b` selects the exact pre-frozen grouped bytes under the registered developer-correctness, regression, resource and no-new-failure gates. All other frozen inputs, source, models and the six consumption/completion receipt pairs remain unchanged. The verifier now intentionally rejects the promoted selected config because its pre-selection baseline must be OFF; it was not relaxed, and no validation budget was reset. See [the selection receipt](cycle2-selection.json).

The final-source real-catalog replay retains all 24 calls and all controls, with no verified real-catalog grouped intervention. The clearly invented cotton/linen example verifies a penalty reduction and rank 2 to 1 under the same query and retrieved candidates. Healthy, missing-model and invalid-weights full-catalog probes also complete under OS network denial, preserving corrections and valid unique IDs. These are demonstrations and operational checks, not independent shopper outcomes.

A read-only final evidence audit recomputed official aggregate metrics, token totals, diagnostics and authored assertion/group counts from raw results. It verified nine target runs (2,487 recorded turns), six capability runs (120 turns), all run-file hashes, unchanged source/model inventories, exact selected config, the 737-turn OFF parity, 12 validation ledger files, 24 replay contracts, six offline contracts, report resource-table values and local document links. No model inference or validation rerun was performed during this audit. Raw artifacts and the audit program remain local; [the report](CYCLE2_RESULTS.md) and [aggregate](cycle2-summary.json) carry the release evidence. The README remains zero bytes.

## Intent-routed pipeline evolution

Phases 0-7 of `plan.md` were implemented in isolated commits with configuration gates. The final suite contains 393 passing tests; Ruff and `pip check` pass. The selected configuration passes all 19 private-like assertions with zero fallbacks and reproduces the frozen public result exactly: HitRate@10 0.895000, MRR 0.613746, MTTC 3.245000 and TechnicalScore 0.786724.

No behavior-changing candidate met the predeclared `>= 0.01` promotion threshold. Routed retrieval scored 0.785165, the product guard 0.772674, structured reranking 0.754929, intent-aware clarification 0.753090 and runtime adaptation 0.786636. These features remain opt-in; `configs/selected.json` is unchanged. Phase 8's optional LLM fallback was not activated because the preceding experiments did not justify its additional cost and complexity. Exact phase scope, caveats and interpretation are recorded in [PIPELINE_EVOLUTION_RESULTS.md](PIPELINE_EVOLUTION_RESULTS.md).
