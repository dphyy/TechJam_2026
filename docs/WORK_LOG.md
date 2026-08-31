> **Historical development record — not current release guidance.** Retained to explain implementation progress and earlier experiment decisions. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Implementation log

## 2026-08-26: source pin and first implementation pass

- Canonical checkout: `submission/`, fork commit `9a35be51780ff1caf89eceaabca34259e946f40f`, local branch `feat/track4-evidence-search`.
- Source research preceded implementation. The accepted design centered on reversible preferences, evidence-aware retrieval and controlled evaluation.
- Official evaluator, public labels and catalog remain unchanged. The original starter is retained in `baselines/official.py`.
- Full-public starter reproduced: 200 sessions, HitRate@10 0.125, MRR 0.068034, MTTC 9.81, Efficiency 0.119, TechnicalScore 0.106710. Exactly matches the pinned published reference.
- Frozen target-disjoint scenario-stratified split: development 160, reserved 40, seed 20260826. Protocol is in `EXPERIMENT_PROTOCOL.md`. Reserved outcomes have not been opened.
- Development starter: HitRate@10 0.13125, MRR 0.074365, MTTC 9.74375, TechnicalScore 0.113059. This is a different subset; do not compare it directly to all-public results.
- Machine: Apple arm64, 16 GiB unified RAM, 10 CPU cores. Python 3.12.12 environment; Torch 2.8.0 verifies MPS availability.
- Pinned BGE-small-en-v1.5 and MiniLM-L6-v2 model assets downloaded and checksum-verified. No paid APIs or compute.
- Added catalog evidence, strict config, reversible state, field-weighted sparse retrieval, evidence ranking, offline neural plumbing, uniform neighbor-contrast compilation, and evaluation-only instrumentation.
- Test-first unit evidence: missing-module red tests followed by passing catalog/config/split/retrieval/ranking/model/contrast tests. Conversation state: 41 focused tests pass.
- Agent integration and all five question / three slate policies completed. Root verification reached 133 unit/integration tests. Runtime failure cases include absent models, raised inference timeouts and malformed ranking returns.
- Uniform BGE index: 50,000 × 384 float32, 302.47 seconds build on MPS, process peak RSS 849,035,264 bytes. Runtime timing runs were not overlapped with preprocessing.
- First sparse/evidence/ledger/other development run: HitRate@10 0.68125, MRR 0.306967, MTTC 5.10625, Efficiency 0.589375, TechnicalScore 0.550590; zero fallbacks/errors. Warm p95 0.2375 seconds, cold start 14.11 seconds, full evaluator process peak RSS 578,928,640 bytes. Sources did not change during the run.
- Paired TechnicalScore improvement over the development starter: +0.43753, bootstrap 95% interval [0.36879, 0.50683]. This is development evidence, not private-test prediction.
- Development traces show high candidate recall but ranking loss; semantic reranking and same-backbone ablations are next. Reserved outcomes remain unopened.
- Product-family diagnostic: zero reserved targets share a normalized exact title or a color/number-stripped title with development targets. This is only a heuristic overlap check, not proof of family independence.

## 2026-08-26: ablations, review and bounded tuning

- Completed sparse/dense/neural/evidence controls and same-backbone state, question, contrast and slate ablations. Exact outcomes and negative results are in `TUNING_DECISIONS.md` and the report.
- Completed simplification, seven local review scopes and independent finding validation. Four validated issues were fixed. An external cross-model review could not start because its CLI rejected an option; no external review coverage is claimed.
- Installed the full transitive dependency lock in a fresh `.venv-repro` and passed `pip check`. Verified both actual models with OS-level networking denied and named provider-token variables removed.
- Iteration 3 tested two additional question caps and two reranker weights. Four questions matched nine; higher neural weight justified the predeclared wider-prefix test. Reserved outcomes remained unopened.
- A development-only failure audit caught lexical no-show negation, avoidance-instruction evidence, metadiscourse-as-color and broad neutral-other retraction bugs. Added invented-data regressions and corrected them. Root verification passed 223 tests, Ruff and whitespace checks.
- Corrected sparse control: 0.699945 TechnicalScore, 0.81250 hit rate. Corrected 30-candidate reranker at weight 0.75: 0.775118 score, 0.88750 hit rate. These are development-only measurements; final confirmation and source freeze follow.
- No model training, learned calibration, public-target lookup, evaluator change or reserved-outcome tuning occurred.

## 2026-08-26: frozen local release

- Final source/config freeze: 08:54:43 UTC / 16:54:43 SGT. Each of the two finalists evaluated once on 40 reserved public sessions. Both hit 37/40; the 30-prefix system had score 0.833146 versus 0.832833 for 60, with approximately half the neural work. The provisional 30-prefix selected config remained unchanged.
- No runtime or parameter edits after reserve. Selected whole-public official CLI, fresh environment, OS networking denied: 179/200 hits, MRR 0.613746, MTTC 3.245, Efficiency 0.775500, score 0.786724. All 200 session records exactly match the frozen split-run union.
- Starter reverified under OS network denial: 25/200 hits and score 0.106710. Selected whole-command wall time 148.32s, maximum RSS 1,241,497,600 bytes, local input tokens 2,375,727 and zero completion tokens.
- Generated `artifacts/demo-final/` with actual responses, transcript, manifest and a three-minute narration-paced `.cast`. Soft-preference/accessory ranking limitations are documented; there is no MP4 or public upload.
- Verified actual missing-model fallback under OS network denial using the full catalog: ten unique valid IDs and zero model tokens. Model absence does not reproduce neural accuracy.
- Saved `docs/experiment-summary.json` with 32 completed run aggregates, eight paired comparisons, full finalist source freeze and model/index provenance. README, report, local Devpost text and release checklist are complete for local handoff. Team/license/host/build-window/video/publication items remain explicit external gates.

## Authority and preparation record

At the frozen evaluation handoff, all changes were local and uncommitted; no remote mutation, submission, outreach or paid service had been used. Preparation/build activity is dated 2026-08-26. The team must confirm organizer eligibility/build-window treatment before competition submission. No claim of private-test performance or prize outcome is made.

## 2026-08-26: source publication preparation

The project owner authorized source publication to the existing fork, with 15–20 atomic commits. The release is organized into 19 dependency-ordered commits using current timestamps and the owner's configured identity; this is packaging of already completed work, not a reconstructed historical development timeline. Runtime, test and experiment sources remain identical to the finalist freeze. Documentation adds a read-only score-ceiling calculation and distinguishes repository publication from video upload and competition submission.

Publication target: [feat/track4-evidence-search](https://github.com/SaaiAravindhRaja/techjam-conversational-search/tree/feat/track4-evidence-search). Downloaded catalogs, models, indexes, raw traces, environments and private working materials are excluded. The original participant-kit public fixtures remain unchanged.
