# Release and external gates

This is a preparation build dated 26 August 2026, not a submitted entry or a claim of prize eligibility. Do not publish catalog data, model files, evaluation traces or credentials merely because they exist locally.

## Local verification

- [x] Pin upstream `9a35be51780ff1caf89eceaabca34259e946f40f`; preserve the official evaluator and original data.
- [x] Reproduce the original starter's published 200-session score exactly.
- [x] Register a target-disjoint 160/40 development/reserved protocol before tuning.
- [x] Implement catalog validation, reversible state, bounded sparse/dense retrieval, semantic reranking and grounded contrast experiments.
- [x] Implement and test simple, entropy and rank-aware question policies plus independent slate policies.
- [x] Complete same-backbone ablations, freeze finalists, and evaluate the reserved subset once per finalist. Select the 30-prefix system; no post-reserve tuning.
- [x] Complete the original formal review and verify resolved findings; 223 tests and Ruff passed at that checkpoint. External cross-model review was unavailable, as disclosed in `QUALITY_LOG.md`.
- [x] Verify the selected release in a fresh environment with networking denied, and exercise the real missing-model fallback with the full 50,000-product catalog.
- [x] Finish the original measured report, aggregate evidence, local Devpost draft and real demo replay. The terminal recording is not a YouTube-ready video.
- [x] Complete cycle-2 three-control development and one-shot locked validation after review fixes; select the exact frozen grouped config under the registered no-regression/resource gates. All 347 tests pass. Target scores and locked capability passes do not improve over frozen behavior.
- [x] Preserve all five capability-validation failures, the absence of a real-catalog grouped intervention, source/config/consumption receipts and actual offline fault evidence in [the cycle-2 report](CYCLE2_RESULTS.md) and [aggregate](cycle2-summary.json).
- [x] Produce the final 24-call alternatives comparison replay and [recording guide](DEMO_SCRIPT.md), clearly separating real catalog exchanges from the invented correctness witness. Encoding and public upload remain pending.
- [x] Restore the landing README with current setup, evaluation, caveats and the judge-showcase command.
- [x] Merge confidence-gated intent, semantic reset and repeat-driven unseen paging; reproduce the selected release at 0.844994 with zero fallbacks and agent errors.
- [x] Add bounded mixed review priors; public score 0.866792, Cycle 5 and Cycle 3 comparisons, constraint ablations, and remaining negation-scope limitations documented in REVIEW_PRIOR_RESULTS.md.
- [x] Reject unguarded early paging after its intent-override regression, then promote a separately registered override-reset guard under the TechnicalScore non-decline rule.
- [x] Screen preregistered low-margin neural fusion; reject it after MRR and TechnicalScore declined, preserving the receipts and disabled implementation.

## Organizer decisions still required

1. Confirm the final evaluator release and how TechnicalScore maps to human judging. The early track brief and general event rules describe different rubric weights.
2. Confirm CPU/GPU, memory, disk, cold-start and per-turn/session limits. Laptop measurements are not proof of final-host compliance.
3. Confirm whether catalog-derived indexes and bundled pretrained weights are allowed, their size limit, and whether final evaluation has network access. This implementation must have a tested offline core.
4. Confirm short/empty recommendation slate treatment. Keep the release at Top-10 unless a verified rule permits a different choice.
5. Confirm training/calibration policy. No model training or hidden-label calibration is used here.
6. Confirm data redistribution/deletion scope, including derived indexes. Source attribution is not blanket permission to redistribute data. Do not delete originals or derived data without a specific approved target list.
7. Confirm treatment of work prepared before the official 29 August noon to 1 September noon SGT window. Preserve this dated record and document significant competition-period contributions separately.

Questions are drafted, not sent. See the [score and judging guide](SCORING_AND_JUDGING.md) for the scoring explanation and rubric discrepancy. Official sources: [event rules](https://tiktoktechjam2026.devpost.com/rules), [participant submission rules](https://github.com/TechJam2026/techjam-conversational-search/blob/9a35be51780ff1caf89eceaabca34259e946f40f/docs/submission_rules.md).

## Packaging and attribution

- Keep original catalog/evaluator/public data hashes in the report and machine-readable manifests.
- Retain code and model license notices. See `DATA_ATTRIBUTION.md` and `MODELS.md`; inspect actual licenses before redistribution.
- Keep downloaded catalogs, all evaluation traces, virtual environments and model/index caches out of source control.
- Supply selected config, dependency manifests, exact commands and model acquisition instructions. State explicitly which files a restricted-network judge needs before execution.
- If installation is also air-gapped, arrange preinstalled pinned dependencies or a target-platform wheelhouse. Verified network denial covers post-install execution; a fresh offline wheelhouse install has not been tested.
- Keep optional research modules disabled unless their measured benefit justifies runtime cost.
- Record team members and actual contributions; do not invent teammates or contributions.

## Publication history and current boundary

- The original preparation branch was published to the public fork on 26 August 2026. That historical exposure is not erased by making a separate private repository.
- The current personal repository remote is `dphyy/TechJam_2026`, with local work on `main`. The older Mercury repository remains configured as `upstream` for provenance.
- The owner authorized verified, atomic private checkpoints and integration into `main`. Merge `fabc7ed` preserves all 32 existing implementation and evidence commits without rewriting them. Use normal history-preserving merges for pull requests, never squash the work into one commit.
- Use the configured owner identity, natural commit subjects whose first word is capitalized, and explicit file staging. Avoid conventional prefixes such as `docs:` and `feat:`. Keep each commit meaningful; do not pad the history. Do not include private datasets, traces, model caches, credentials or planning-only files.

## Remaining submission actions

- [ ] Approve a final public source location separately. The private development branch is not an eligible public submission URL.
- [ ] Confirm eligible significant competition-period work and complete the dated contribution record.
- [ ] Produce/export the required three-minute video and approve the public YouTube upload.
- [ ] Put the approved public source URL in the Devpost draft when publication is authorized.
- [ ] Fill the public video URL in the Devpost draft after its separate approved upload.
- [ ] Approve and complete Devpost submission before the verified deadline.

Private checkpoint and merge authorization does not authorize a public release, video upload, outreach, competition submission, paid inference or rented compute. Verify `origin/main`'s exact commit after pushing; a prepared link alone is not evidence of publication.
