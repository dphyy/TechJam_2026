# Mercury: measured conversational search

Preparation build, 26 August 2026. Not submitted to the competition. This report distinguishes measured public-session results from unknown private-test performance and does not claim a prize outcome or real purchase uplift.

## Current release addendum — 30 August 2026

The historical frozen-selection study below remains intact, but it no longer describes the current source exactly. The selected 30-candidate MiniLM pipeline now pages an unchanged ranking from turn 5 and recognizes a broader set of direct corrections, replacement phrases, generic slate rejection, and excess-feedback language. The parser changes are covered by invented-data regression tests rather than public target-specific rules.

| Current selected release | n | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|---:|
| Released public development set | 200 | 0.970000 | 0.645919 | 2.980000 | 0.802000 | 0.839176 |

The 30 August reproduction recorded 2,375,969 local prompt tokens, p95 turn latency of 0.555 seconds, zero fallback turns and zero agent-error turns. Its source/config/data receipts are in `runs/hardened-selected-public-20260830/`, with the aggregate in [current-results.json](docs/current-results.json). Because these 200 sessions have been used throughout development, this is an implementation reproduction—not a private-set prediction.

A preregistered low-margin neural-fusion candidate was rejected on a separate 160-session popularity-matched Cycle 5 screening pack: hit rate tied, while MRR fell by 0.005546 and TechnicalScore fell by 0.003038. It remains disabled. See [the screening result](docs/MARGIN_FUSION_RESULTS.md).

For presentation, `python -m demo.showcase --results docs/current-results.json --output artifacts/judge-showcase` creates an inspectable HTML report from real agent calls. It exposes corrections, retained/retracted state, routing, paging, fallbacks, catalog IDs and conservative evidence labels. Missing catalog evidence remains unknown.

## Method and product

Mercury is a backend shopping copilot that can revise its understanding without losing useful context. A user can switch from black leather to blue canvas while retaining a shoulder-bag and zipper requirement. A source-linked preference ledger retracts affected assertions; broad lexical retrieval keeps recall; explicit contradictions are demoted while missing metadata stays unknown. A compact, local cross-encoder can reorder the strongest candidates. Recommendations are real, unique catalog IDs, optionally accompanied by one allowed follow-up question.

The released interface is the kit's `Agent(catalog_path)`, `reset` and `respond`, with a fixed Top-10 slate. There is no shopping frontend, hosted service, live shopping transaction or text-generating model. See [README](README.md) for commands and [design](docs/DESIGN.md) for interfaces and limits.

The engineering contribution is the combination of reversible state, provenance-aware unknown handling, controlled experiments and offline reliability. BM25, cross-encoders, conversational memory and contrastive retrieval are established techniques; no world-first claim is made.

## Inputs and evaluation integrity

- Pinned participant source: `9a35be51780ff1caf89eceaabca34259e946f40f`. The official evaluator and scoring configuration are unchanged.
- Catalog: 50,000 distinct product IDs, SHA-256 `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.
- Public sessions: 200, SHA-256 `857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`.
- Before tuning, freeze a target-disjoint, scenario-stratified 160/40 development/reserved split, seed 20260826. Development has 64 buying, 64 browsing, 24 override and 8 boundary sessions; reserved has 16/16/6/2.
- Zero cross-split target overlap and zero overlap in normalized exact-title or color/number-stripped title families. The latter is a heuristic, not proof of manufacturer-family independence.
- All 200 public rows had received aggregate inspection and a starter run before splitting. The reserve is held-out public development, not secret/private data.
- Runtime modules do not read labels, public sample IDs, scenario flags or simulator internals. Evaluator-side diagnostics join targets only after inference. No target lookup tables, profile fingerprinting, purchase-history joins, model training or learned calibration are used.

The unchanged score is `0.50 × HitRate@10 + 0.30 × MRR + 0.20 × clip((11 − MTTC)/10, 0, 1)`. Failed sessions contribute an MTTC of 11. First-hit and override-gate semantics come from the official evaluator, not a rewritten approximation.

Local `runs/` directories retain results, traces and manifests. Iteration 2 onward also archives the exact source tree used by each run. Earlier exploratory runs retain source hashes but do not all have complete source snapshots; they are not presented as the frozen reproducibility evidence. Only aggregate results and release provenance belong in the public-facing evidence package, subject to organizer data terms.

[Machine-readable evidence](docs/experiment-summary.json) contains 32 run aggregates, eight paired comparisons, the complete finalist source freeze, model/index provenance and the offline official-run summary. Raw labels and session traces are not included in that aggregate artifact.

[Protocol](docs/EXPERIMENT_PROTOCOL.md) predeclares a 0.01 practical score gain, maximum 0.02 hit-rate loss, resource caps, at most two frozen finalists and one reserved run per finalist. Confidence intervals use 10,000 paired session bootstrap resamples, seed 20260826. Development comparisons are exploratory and not adjusted for repeated testing. No tuning is allowed after reserved evaluation.

## Reproduced starter

| Dataset | n | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|---:|
| Whole public, original starter | 200 | 0.125000 | 0.068034 | 9.81000 | 0.119000 | 0.106710 |
| Development, original starter | 160 | 0.131250 | 0.074365 | 9.74375 | 0.125625 | 0.113059 |

The whole-public starter exactly reproduces the pinned published reference. These two rows have different samples and must not be compared as an improvement.

## Development experiments and negative results

Iteration 1 built a complete sparse agent first (score 0.550590), then added local reranking (0.591873). Iteration 2 improved generic intent parsing and cache handling. Its same-backbone controls follow; all use the same 160 development sessions, and no source changed during a run.

| Iteration-2 configuration | HitRate@10 | TechnicalScore | Interpretation |
|---|---:|---:|---|
| Sparse, optional positive-evidence boosts | 0.75000 | 0.610760 | Extra boosts hurt ranking |
| Sparse, boosts off, ledger + simple questions | 0.78750 | 0.677612 | Ordinary stateful control |
| Add local reranker | 0.81250 | 0.706863 | +0.02925; paired 95% CI [0.01234, 0.04926] |
| Add dense retrieval instead | 0.76875 | 0.645400 | Lower score and extra model/index cost |
| Dense + reranker | 0.80000 | 0.685234 | Worse than sparse + reranker |
| Latest-per-attribute state | 0.63125 | 0.547427 | Loses useful context |
| Unretracted history state | 0.80000 | 0.688393 | Slight score gain, but retains withdrawn requirements; not release-correct |
| Answer-aware attribute schedule | 0.79375 | 0.642476 | Slower target recovery |
| Entropy questions | 0.80000 | 0.665713 | Did not beat simple questions |
| Rank-value questions | 0.80625 | 0.673681 | Did not beat simple questions |
| Catalog-neighbor contrast | 0.80625 | 0.679245 | +0.00163, below practical threshold |
| Gap / short-lookahead slates | 0.78750 | 0.677612 | Neither changed outcomes |
| No questions | 0.19375 | 0.165118 | Follow-up information matters greatly here |

The questions change future simulator disclosures, so these are end-to-end dialogue comparisons, not controlled fixed-query retrieval tests. The rank-value policy includes unknown/no-preference answers and outside-pool uncertainty, but its weights are heuristic and the small public sample does not validate calibration.

Iteration 3 adds independently reviewed correctness fixes: direct-negation-aware evidence, clause-ordered corrections, input-bound consistency, official-hit diagnostics and an always-on unknown-safe contradiction guard. Earlier higher scores are not a reason to revert correctness. Positive boosts remain optional; the guard is separate. The bounded final search and every negative result are recorded in [tuning decisions](docs/TUNING_DECISIONS.md).

The final development correction pass also addressed lexical no-show phrases, direct avoidance instructions, metadiscourse masquerading as a color and unanswered broad questions erasing earlier details. These are general rules with invented-data regressions, not public-target special cases. Iteration 4 results on the stable corrected source:

| Configuration, simple questions capped at four | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Sparse, no optional positive boosts | 0.81250 | 0.513983 | 4.02500 | 0.697500 | 0.699945 |
| Reranker weight 0.50, 30 candidates | 0.87500 | 0.558361 | 3.40625 | 0.759375 | 0.756883 |
| Reranker weight 0.75, 30 candidates | 0.88750 | 0.589978 | 3.28125 | 0.771875 | 0.775118 |
| Reranker weight 0.75, 60 candidates | 0.90625 | 0.592904 | 3.14375 | 0.785625 | 0.788121 |

The 30-candidate neural system improves over the corrected sparse control by +0.07517, paired 95% CI [0.03985, 0.11120], with +0.075 hit rate. Weight 0.75 versus 0.50 gains +0.01824 but CI [-0.00006, 0.03749] is inconclusive. Moving from 30 to 60 candidates gains +0.01300, CI [-0.00256, 0.03142], while approximately doubling neural work. These last two comparisons do not establish certain superiority. The highest-scoring ordinary system is the backbone for the final research-module tests.

Final same-backbone tests on the 60-candidate system: contrast reaches 0.793321 score / 0.91250 hit rate, but its +0.00520 score gain has CI [-0.00497, 0.01771] and misses the practical threshold. Rank-value reaches 0.761205 score / 0.90625 hit rate: a -0.02692 score delta, CI [-0.03952, -0.01423], with slower MTTC (4.10 versus 3.14375). Both modules stay disabled. The strongest defensible system is an ordinary retrieval/reranking backend with careful reversible state, not a claim that either experimental novelty won its ablation.

## Frozen selection and final evaluation

Source and both finalist configurations were frozen at **2026-08-26 08:54:43 UTC**, before reserved evaluation. Each was then evaluated once on the 40-session reserve. No source or parameter tuning followed.

| Frozen configuration / dataset | n | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|---:|
| Selected 30-prefix / development | 160 | 0.88750 | 0.589978 | 3.28125 | 0.771875 | 0.775118 |
| Selected 30-prefix / reserved public | 40 | 0.92500 | 0.708819 | 3.10000 | 0.790000 | 0.833146 |
| Alternative 60-prefix / reserved public | 40 | 0.92500 | 0.709444 | 3.12500 | 0.787500 | 0.832833 |
| Selected / whole public, OS-offline reproduction | 200 | 0.89500 | 0.613746 | 3.24500 | 0.775500 | 0.786724 |

Both finalists hit 37 of 40 reserved targets. The 60-prefix minus 30-prefix paired score difference is -0.0003125, 95% CI [-0.0009375, 0.0000000], with zero hit-rate difference. Select the **30-candidate** system: no measured reserve benefit from doubling reranking work. This is selection between pre-frozen candidates, not a post-reserve parameter change.

[Selected configuration](configs/selected.json): field-weighted sparse retrieval, 120 retained candidates, 30-candidate local MiniLM reranking at weight 0.75, reversible ledger, mandatory unknown-safe contradiction guard, simple questions capped at four and fixed Top-10 slates. Dense retrieval, optional positive evidence boosts, contrasts and adaptive question/slate policies are disabled. [Sparse fallback](configs/sparse_fallback.json) retains the same state, constraints and dialogue policy without the model.

| Selected scenario | Development n / hit / MRR / MTTC | Reserved n / hit / MRR / MTTC |
|---|---|---|
| Buying | 64 / 0.859375 / 0.540340 / 3.00000 | 16 / 0.937500 / 0.651215 / 2.37500 |
| Browsing | 64 / 0.921875 / 0.591071 / 2.87500 | 16 / 0.875000 / 0.693750 / 3.06250 |
| Intent override | 24 / 0.833333 / 0.713542 / 4.791667 | 6 / 1.000000 / 0.805556 / 3.833333 |
| Boundary | 8 / 1.000000 / 0.607639 / 4.25000 | 2 / 1.000000 / 1.000000 / 7.00000 |

Small scenario slices, particularly two reserved boundary sessions, do not establish broad robustness. A whole-public offline reproduction follows selection and is descriptive only, not a second independent holdout.

The whole-public run used the actual selected `agent.py` entry through the unchanged official CLI in the fresh environment, under OS network denial with named provider credentials removed. All 200 per-session results exactly match the union of the frozen development and reserved runs. It found 179/200 targets versus the reproduced starter's 25/200. Whole-public TechnicalScore is 0.786724 versus 0.106710; this large weak-baseline gap must not obscure the smaller, more meaningful comparisons against strong ordinary retrieval/reranking controls above.

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  .venv-repro/bin/python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl \
  --output artifacts/official-final-offline.json
```

`sandbox-exec` is a macOS verification tool, not a runtime dependency or a privileged-host requirement for the agent. The command returned exit 0. `/usr/bin/time -l` measured 148.32 seconds whole-command wall time and 1,241,497,600 bytes maximum resident size. Total local input tokens: 2,375,727; completion tokens: zero. The selected entry still requires the prefetched reranker files for neural reproduction.

## Reliability, resources and cost

All inference runs use an existing local Apple M4 (arm64), 16 GiB unified memory and 10 CPU cores. CPU model inference uses four threads. CPython 3.12.12 and the full transitive dependency lock were installed in a fresh virtual environment; `pip check`, 223 tests and Ruff passed before iteration 4. No static type-checker is configured.

Both pinned models completed actual inference under OS-level network denial with named provider credentials removed. A separate socket probe verified the denial. The full selected-entrypoint reproduction above also passed under OS network denial. All reported token counts are actual local model input tokenization, not paid API tokens; completion tokens are zero. Paid API/compute cost is US$0, excluding existing hardware, electricity and internet access.

| Selected run | Agent cold start | p50 / p95 response | Peak process RSS, bytes | Local input tokens | Fallbacks |
|---|---:|---:|---:|---:|---:|
| Development, 507 turns | 16.249s | 0.260s / 0.351s | 1,207,205,888 | 1,894,530 | 0 |
| Reserved, 121 turns | 18.716s | 0.278s / 0.388s | 940,900,352 | 481,197 | 0 |

`cold_start_seconds` measures agent import/construction in a fresh process, excluding interpreter startup, source snapshotting and evaluator dataset loading; the operating system's file cache may be warm. Turn latency includes both changed and cached intent (140/507 development turns and 30/121 reserved turns were cached). Memory includes the evaluator process, not isolated per-agent memory. The engineering caps are not organizer-approved limits. The selected runtime needs only the approximately 88 MiB reranker assets; BGE vectors and contrast sidecars are not required.

Optional preprocessing was uniform over all products:

- BGE dense index: 50,000 × 384 float32, approximately 74 MiB including IDs. Build: 302.47 seconds on MPS, 849,035,264 bytes process peak RSS; this does not measure all unified GPU allocation.
- Lexical neighbor sidecar: eight neighbors, six rare title terms, at most 256 postings per term. Final `lexical-neighbors-v2-negation` build: 90.63 seconds excluding catalog loading, 104.20 seconds whole command, 378,978,304 bytes peak process RSS and 40,316,285 bytes of contrast data. SHA-256 `28552142ab4ae1579c1dcc2f8d5afb339301f8e5e76a5b92c0e1d445a6ee5fdf`. The initial v1 sidecar (80.82 seconds, 40,314,096 bytes) is retained locally under `artifacts/contrast-v1`, not overwritten.
- Model files: approximately 128 MiB BGE and 88 MiB MiniLM on disk. The dense module is optional; the reranker does not require the BGE model or vectors. Exact pinned revisions, hashes and licenses are in [models](docs/MODELS.md) and local asset manifests.
- Sparse FTS5 is built in memory at startup; it has no persistent index artifact or external database. Repeated unchanged intent reuses its ranking.

## Failure analysis and limitations

Selected ever-ranked target recall at depths 10 / 30 / 60 / 120: development 0.8875 / 0.9000 / 0.9625 / 1.0000; reserved 0.9250 / 0.9750 / 1.0000 / 1.0000. All targets were retrieved at some point; the 18 development and 3 reserved official misses therefore fall in the coarse ranking/dialogue bucket, with zero agent-error turns. Recall is measured across a policy-dependent session, not a fixed-turn or independent-query benchmark. These diagnostics cannot causally separate every parser, ranking and question error. The final development audit's concrete false-exclusion and unintended-retraction cases were fixed before freeze; no such reserved cases were used to tune code.

Catalog ambiguity is intrinsic: the initial audit found ten pairs identical across all exposed catalog fields. IDs are preserved, and the system invents no distinction. Approximately 79.17% of products lack a usable price; missing price is unknown, and a lower-bound price does not prove affordability. Text evidence may itself be wrong or contradictory. The English rule parser handles explicit corrections and tested paraphrases, not arbitrary multilingual or implicit intent.

Reliability tests cover missing/corrupt assets, malformed ranking output, raised inference timeouts, cache provenance, session reset/isolation/eviction and final-turn legality. A permanently stalled native model kernel is not interrupted by an in-process watchdog; final host timeouts still need confirmation. Missing optional models produce a legal sparse fallback, not an assertion of identical neural accuracy.

Actual fallback verification used the selected neural config with an empty temporary asset directory and the full 50,000-product catalog under OS network denial. It recorded `FileNotFoundError` for the reranker, returned ten unique valid IDs and reported zero model tokens. A socket probe confirmed network denial. The real demo outputs and three-minute terminal recording are under `artifacts/demo-final/`; see the [demo guide](docs/DEMO_SCRIPT.md) for reproduction and honest narration of remaining ranking limitations.

Uninformative replies intentionally reuse the previous ranking; when ranking is wrong and no useful new preference arrives, a dialogue can stall. Explicit negative feedback and richer open-vocabulary corrections need broader independent testing. A larger private or real-user evaluation would be needed to validate ranking, question calibration and transfer beyond this public simulator.

Seven local review scopes and independent finding validation completed; four validated issues were fixed. A requested external cross-model review failed at CLI startup twice, so no external review assurance is claimed. See [quality log](docs/QUALITY_LOG.md).

## Submission and remaining external gates

The [pinned submission rules](https://github.com/TechJam2026/techjam-conversational-search/blob/9a35be51780ff1caf89eceaabca34259e946f40f/docs/submission_rules.md) require a Python entry file/helpers, setup, method/model/limitations report and resource/token/cost disclosure. The general event also requires public submission media; a backend track does not make presentation quality irrelevant.

This is pre-window preparation. Confirm eligible significant updates during the 29 August noon–1 September noon SGT build window and retain the dated contribution record. Ask organizers about final runtime/asset limits, index/model packaging, training and data redistribution/deletion requirements, and the mapping between TechnicalScore and human judging. Do not treat uncertainty as permission.

Devpost text and a real three-minute terminal replay are preparation assets. A YouTube-ready video still requires recording/export, rights review and approved upload. Source publication is authorized separately from video upload or competition submission; neither of those latter actions has occurred. Team roster/contributions and the final video URL require confirmation. See the [release checklist](docs/RELEASE_CHECKLIST.md) and [score and judging guide](docs/SCORING_AND_JUDGING.md).
