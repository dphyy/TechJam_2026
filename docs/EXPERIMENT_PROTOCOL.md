> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Experiment protocol

Registered before candidate tuning on 26 August 2026. This is early preparation; competition-period contributions must be recorded separately.

## Frozen inputs and isolation

- Upstream source commit: `9a35be51780ff1caf89eceaabca34259e946f40f`.
- Catalog SHA-256: `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.
- Public-set SHA-256: `857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`.
- Keep the official evaluator, configuration and public labels unchanged. `baselines/official.py` preserves the unmodified starter for later reproduction.
- The 200 public rows have already received aggregate inspection and the official starter has been run. Reserved data is held-out public development, not secret private-test evidence.
- Partition by target ID; use deterministic hash ordering and scenario stratification. Target sizes are 160 development / 40 reserved, adjusted if target grouping requires it. Produce a split manifest and never send it or labels to runtime code.
- Tune only on development data, with four development-only folds for any learned calibration if organizer policy permits it. Preserve all catalog products in indexes; no public-target special casing.

## Candidate selection

Start with stateful field-weighted BM25 plus useful `other` questions. Compare an answer-aware schedule; add dense retrieval/reranking, evidence state, contrast compilation, and policy changes one at a time on a shared backbone.

The primary metric is TechnicalScore. A meaningful development improvement target is at least 0.01 absolute, with no more than 0.02 absolute loss in HitRate@10. These are engineering selection thresholds, not claims of statistical significance. Prefer the simpler method if gains are smaller than uncertainty or resource costs are disproportionate.

Choose at most two frozen contenders on development results before opening reserved results: the strongest simple policy and the strongest justified enhanced policy. Evaluate each once on the reserved subset, report paired per-session deltas and a 95% paired bootstrap interval (10,000 resamples, seed 20260826), and do not retune from those outcomes. If uncertainty is wide, label it inconclusive. A later whole-public run is descriptive only.

## Resource budget and boundaries

Paid-service budget is zero. No remote inference, cloud compute, or hosted vector database. The local machine is arm64 with 16 GiB unified RAM and 10 CPU cores. Initial engineering budgets, not organizer limits: agent peak RSS under 6 GiB; warm p95 response under 3 seconds for the sparse agent and under 10 seconds for a neural configuration; index/model assets under 2 GiB. Measure and revise these targets with reasons before final selection, never relabel them as official requirements.

Use fixed local model revisions and clean-environment reproduction. Do not run competing benchmarks concurrently. Capture wall time, peak memory, cold start, actual tokens, fallback counts, and index/model hashes. The sparse path must operate without models, network, or credentials.

## Interpretation

Report HitRate, MRR, MTTC, Efficiency and TechnicalScore with sample counts and scenario slices. Small Boundary samples have high uncertainty. Separate retrieval misses, ranking mistakes, stale-state errors, false exclusions and irreducible catalog ambiguity. Official unchanged-harness results and synthetic/paraphrase robustness checks stay separate.

No simulated metric implies real purchase conversion uplift. Rank-value probabilities remain heuristic unless calibrated and validated. Current short-slate legality and final network/hardware limits need organizer confirmation; the conservative release remains Top-10 plus a useful question. Failed modules remain documented experiments, not mandatory production dependencies.
