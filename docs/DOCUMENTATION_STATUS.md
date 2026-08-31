# Documentation status and historical exceptions

Audited 1 September 2026. Scope: all project-owned Markdown files and READMEs;
generated outputs, virtual environments, dependency caches and frozen local
receipts are not rewritten. This audit changes documentation only and does not
rerun tests, benchmarks or any held-out dataset.

Coverage: **89 project Markdown files, including all four READMEs**. Local file
links and heading anchors resolve. The public source/result binding and
evaluator/catalog/public-dataset hashes were verified unchanged. Team roles were
checked against the supplied list. Detailed audit receipts remain local under
`output/documentation-audit-20260901/`.

## Current authoritative content

The newest verified public pipeline is **lexical retrieval → adaptive shortlist
→ guarded paging**, exported by `agent.Agent` / `starter.agent.Agent` /
`mercury.lexical.Agent` with `DEFAULT_AGENT_CONFIG`. It requires Python 3.10+
with SQLite FTS5, no model and no third-party runtime packages. The recorded
verification used Python 3.13.5.

The current public result is **200/200 targets, TechnicalScore 0.967414,
MRR 0.965048, MTTC 2.105**, with zero model tokens/errors/fallbacks. The latest
runtime verification contains **1,164 full-suite tests** and **130 no-site-packages
checks**. These are the existing source-bound measurements, not new measurements
from this documentation audit. Exact provenance remains in
[current-results.json](current-results.json) and [verification](PIPELINE_VERIFICATION.md).

| README checked | Current purpose |
|---|---|
| [Project README](../README.md) | Overview, setup, exact reproduction, newest metrics, limits, future work and supplied team roles |
| [Data README](../data/README.md) | Current catalog/public-set checksums, preparation and consumed-data boundaries |
| [Configuration README](../configs/README.md) | Public configuration in Python; JSON configurations are optional historical research |
| [History README](history/README.md) | Redirect policy and intentional comparison artifacts; no old setup recipe |

[Setup](SETUP.md), [design](DESIGN.md), [demo](DEMO_SCRIPT.md),
[submission draft](DEVPOST_DRAFT.md), [submission write-up](DEVPOST_WRITEUP.md),
[release checklist](RELEASE_CHECKLIST.md), [scoring guide](SCORING_AND_JUDGING.md),
[rubric context](CHALLENGE_RUBRIC_CONTEXT.md), and [maintenance status](../plan.md)
now agree with those references. Public results are consumed development evidence,
not organizer-private accuracy or real-user impact.

## Outdated content removed or corrected

- Replaced the submission write-up's older paging metrics and Python version;
  removed older cross-set/repetition results from the current submission narrative.
- Removed the write-up's stale “freeze and re-evaluate next” task: that verification
  is complete, while external release actions remain unverified.
- Corrected the optional-model guide to `requirements-research.txt`; removed the
  old environment validation claim as a description of the current runtime.
- Added the already recorded robustness-v1 final consumption to dataset status.
  No old sealed filename or protocol can restore its holdout status.
- Updated the authored-validation instructions to run the public lexical checks,
  not the older neural `selected.json` capability runner.
- Replaced the nine `history/*-before-lexical-paging.md` bodies with redirects;
  obsolete setup, model requirements, scores and submission instructions no longer
  remain in duplicate guides. Their paths stay valid for existing links.
- Changed missing-team placeholders to the five supplied names/roles. A dated
  contribution record and external submission evidence still require real facts.

## Where older results intentionally remain

These exceptions explain improvements, regressions, rejected methods or the
controls needed to interpret them. They are not claims about the newest runtime.
All listed research records start with a historical-scope notice. Original
experiment scores/test counts are retained instead of relabeled as current.

- [REPORT — improvement history](../REPORT.md#improvement-history-and-evidence-boundaries)
  retains the earlier failing cleanup measurement and the previous frozen paging
  comparison, including the small score decrease after correctness repairs.
- [Verification — repair comparison history](PIPELINE_VERIFICATION.md#repair-comparison-history)
  retains the failed/intermediate/final measurement sequence so the successful
  result does not conceal regressions found during development.
- The explicitly named reports and supporting protocols below preserve the wider
  development history. Keeping negative outcomes is as important as keeping wins.

### Historical comparison (26 files)

- [ADAPTIVE_RERANK_DEPTH_RESULTS.md](ADAPTIVE_RERANK_DEPTH_RESULTS.md)
- [ADMISSION_V2_RESULTS.md](ADMISSION_V2_RESULTS.md)
- [BOUNDED_DIALOGUE_POLICY_RESULTS.md](BOUNDED_DIALOGUE_POLICY_RESULTS.md)
- [CANONICAL_STATE_SEMANTICS_RESULTS.md](CANONICAL_STATE_SEMANTICS_RESULTS.md)
- [CATALOG_VOCABULARY_V2_RESULTS.md](CATALOG_VOCABULARY_V2_RESULTS.md)
- [CONTINUATION_MATRIX_AND_ATTRIBUTION.md](CONTINUATION_MATRIX_AND_ATTRIBUTION.md)
- [CONTINUATION_RELEASE_REPORT.md](CONTINUATION_RELEASE_REPORT.md)
- [CYCLE2_RESULTS.md](CYCLE2_RESULTS.md)
- [CYCLE3B_RERANK_DEPTH_RESULTS.md](CYCLE3B_RERANK_DEPTH_RESULTS.md)
- [CYCLE3C_FULL_POOL_RESULTS.md](CYCLE3C_FULL_POOL_RESULTS.md)
- [CYCLE3_SCREENING_RESULTS.md](CYCLE3_SCREENING_RESULTS.md)
- [CYCLE4_RESULTS.md](CYCLE4_RESULTS.md)
- [CYCLE4_SCREENING_RESULTS.md](CYCLE4_SCREENING_RESULTS.md)
- [CYCLE4_SLATE_REPETITION_FINDING.md](CYCLE4_SLATE_REPETITION_FINDING.md)
- [DOMAIN_RERANKER_RESULTS.md](DOMAIN_RERANKER_RESULTS.md)
- [EARLY_PAGING_OVERRIDE_RESET_RESULTS.md](EARLY_PAGING_OVERRIDE_RESET_RESULTS.md)
- [EARLY_PAGING_RESULTS.md](EARLY_PAGING_RESULTS.md)
- [IDENTICAL_DOCUMENT_GROUPING_FEASIBILITY.md](IDENTICAL_DOCUMENT_GROUPING_FEASIBILITY.md)
- [MARGIN_FUSION_RESULTS.md](MARGIN_FUSION_RESULTS.md)
- [NEURAL_WEIGHT_TUNING_RESULTS.md](NEURAL_WEIGHT_TUNING_RESULTS.md)
- [PHASE15_INTENT_DIAGNOSTICS.md](PHASE15_INTENT_DIAGNOSTICS.md)
- [PIPELINE_EVOLUTION_RESULTS.md](PIPELINE_EVOLUTION_RESULTS.md)
- [PIPELINE_REFINEMENT_RESULTS.md](PIPELINE_REFINEMENT_RESULTS.md)
- [REALISTIC_SHOPPING_MERGE_RESULTS.md](REALISTIC_SHOPPING_MERGE_RESULTS.md)
- [REVIEW_PRIOR_RESULTS.md](REVIEW_PRIOR_RESULTS.md)
- [ROADMAP_IMPLEMENTATION_RESULTS.md](ROADMAP_IMPLEMENTATION_RESULTS.md)

### Historical experiment protocol (25 files)

- [ADAPTIVE_GUARDED_PAGING_PROTOCOL.md](ADAPTIVE_GUARDED_PAGING_PROTOCOL.md)
- [ADMISSION_SCORER_PROTOCOL.md](ADMISSION_SCORER_PROTOCOL.md)
- [CATALOG_VOCABULARY_PROTOCOL.md](CATALOG_VOCABULARY_PROTOCOL.md)
- [CYCLE2_ALTERNATIVES_PROTOCOL.md](CYCLE2_ALTERNATIVES_PROTOCOL.md)
- [CYCLE2_EXPERIMENT_PROTOCOL.md](CYCLE2_EXPERIMENT_PROTOCOL.md)
- [CYCLE3B_RERANK_DEPTH_PROTOCOL.md](CYCLE3B_RERANK_DEPTH_PROTOCOL.md)
- [CYCLE3C_FULL_POOL_PROTOCOL.md](CYCLE3C_FULL_POOL_PROTOCOL.md)
- [CYCLE3_EXPERIMENT_PROTOCOL.md](CYCLE3_EXPERIMENT_PROTOCOL.md)
- [CYCLE4_EXPERIMENT_PROTOCOL.md](CYCLE4_EXPERIMENT_PROTOCOL.md)
- [CYCLE4_RERANKER_SWAP_PROTOCOL.md](CYCLE4_RERANKER_SWAP_PROTOCOL.md)
- [CYCLE4_SLATE_PAGING_PROTOCOL.md](CYCLE4_SLATE_PAGING_PROTOCOL.md)
- [CYCLE5_TARGET_POOL_PROTOCOL.md](CYCLE5_TARGET_POOL_PROTOCOL.md)
- [EARLY_PAGING_OVERRIDE_RESET_PROTOCOL.md](EARLY_PAGING_OVERRIDE_RESET_PROTOCOL.md)
- [EARLY_PAGING_PROTOCOL.md](EARLY_PAGING_PROTOCOL.md)
- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)
- [FRONTIER_RERANK_PROTOCOL.md](FRONTIER_RERANK_PROTOCOL.md)
- [MARGIN_FUSION_PROTOCOL.md](MARGIN_FUSION_PROTOCOL.md)
- [NEURAL_BATCH_THREAD_PROTOCOL.md](NEURAL_BATCH_THREAD_PROTOCOL.md)
- [NEURAL_QUANTIZATION_PROTOCOL.md](NEURAL_QUANTIZATION_PROTOCOL.md)
- [OVERRIDE_TRANSITION_PROTOCOL.md](OVERRIDE_TRANSITION_PROTOCOL.md)
- [PAGE_LOCAL_RERANK_PROTOCOL.md](PAGE_LOCAL_RERANK_PROTOCOL.md)
- [PAIR_LOGIT_CACHE_PROTOCOL.md](PAIR_LOGIT_CACHE_PROTOCOL.md)
- [PRIVATE_ROBUSTNESS_MATRIX_PROTOCOL.md](PRIVATE_ROBUSTNESS_MATRIX_PROTOCOL.md)
- [REVIEW_PRIOR_PROTOCOL.md](REVIEW_PRIOR_PROTOCOL.md)
- [UNSEEN_EVIDENCE_PROTOCOL.md](UNSEEN_EVIDENCE_PROTOCOL.md)

### Historical development record (5 files)

- [CYCLE2_DECISIONS.md](CYCLE2_DECISIONS.md)
- [MERGE_DECISIONS.md](MERGE_DECISIONS.md)
- [QUALITY_LOG.md](QUALITY_LOG.md)
- [TUNING_DECISIONS.md](TUNING_DECISIONS.md)
- [WORK_LOG.md](WORK_LOG.md)

## Other intentional non-current material

- [Optional models](MODELS.md) describes supported legacy experiment assets; it is
  not a requirement for the default pipeline and does not advertise old scores.
- [Proposed shopper study](HUMAN_TEST_PROTOCOL.md) is future work, explicitly not
  conducted. Its controls now refer to the current lexical pipeline.
- [Private-like validation](PRIVATE_LIKE_VALIDATION.md) explains why older research
  runners are not equivalent to current public-agent tests.
- The organizer's [competition specification](competition_specification.md),
  [submission contract](submission_rules.md) and baseline/evaluation JSON files
  are authoritative supplied fixtures, not our current implementation results.
  They are intentionally preserved rather than edited to mirror our agent.
- Frozen local receipts and historical JSON aggregates retain their original
  hashes and outcomes. They are evidence for comparisons, not demo inputs or
  current release metrics. [Dataset status](DATASET_STATUS.md) explains exposure.

The current public runtime, evaluator/data bytes and source-bound result remain
unchanged. No commit, push, publication or submission is performed by this audit.
