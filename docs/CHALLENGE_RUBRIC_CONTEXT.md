# Challenge Rubric and Runtime Alignment

This document is the durable briefing for future work on Mercury. It records the
Track 4 challenge language supplied to the team, maps it to the current codebase,
and prevents completed or rejected experiments from being mistaken for active
work. Read it with [the competition specification](competition_specification.md),
[the scoring guide](SCORING_AND_JUDGING.md), and [the release checklist](RELEASE_CHECKLIST.md).

## Authority and Interpretation

The source brief describes a conversational shopping agent over a frozen Amazon
Reviews 2023 catalog. The goal is exact hidden-product recovery, ranked early and
high, while distinguishing targeted Buying from open-ended Browsing and handling
information accumulation, intent overrides, and no-preference boundaries.

The supplied brief and the general event rules may describe different judging
weights. Preserve the brief weights below for submission planning, but verify the
rules that apply at the actual judging stage. The official evaluator and frozen
participant-kit contract remain authoritative for machine scoring. Never turn
public labels, target IDs, evaluator behavior, or scenario labels into runtime
features.

## Supplied Judging Rubric

| Criterion | Weight | Evidence Mercury should present |
|---|---:|---|
| Technical Execution | 35% | Reliable Agent API, reversible state, typed intent and retrieval plans, legal catalog output, tests, reproducible evaluation, and offline fallback |
| Innovation & Problem Insight | 20% | Buying/Browsing distinction, explicit evidence provenance, intent-override handling, unknown-safe constraints, and honest analysis of failed adaptive experiments |
| Impact & Relevance | 20% | Fewer irrelevant shopping turns, better recovery of changing intent, transparent limitations, and a credible path from exact-item evaluation to real catalog search |
| Feasibility & Practicality | 15% | In-memory execution, bounded local models, fixed compute budgets, no mandatory hosted API, soft handling of unreliable price metadata, and measured latency/cost |
| Presentation & Communication | 10% | A coherent problem-to-evidence narrative, real end-to-end API demonstration, clear score interpretation, limitations, and defensible answers to judge questions |

The evaluator separately reports HitRate@10, MRR, MTTC, Efficiency, and the
combined TechnicalScore. These measurements are technical evidence; they are not
the human rubric itself. See [the scoring guide](SCORING_AND_JUDGING.md) for the
formula, the current `0.786724` selected result, and the rubric discrepancy.

## Challenge Boundaries That Must Persist

- A session has at most 10 turns. Return at most 10 unique, catalog-valid
  `parent_asin` values in ranked order.
- The 50,000-product catalog is read-only. Do not inject mock ASINs into runtime
  output or modify evaluator/catalog artifacts.
- Public development has 200 labeled sessions; the organizer retains 800 private
  sessions with separate users and targets. Public results are development
  evidence, not unseen validation.
- The expected runtime is a lightweight in-memory/headless backend. A storefront
  UI, multimodal processing, full-parameter base-model training, and an external
  industrial vector database are out of scope.
- Local scoring, keyword, dense, hybrid, and reranking approaches are allowed.
  Hosted services are optional, team-funded, secret-safe, disclosed, and may not
  be required for legal offline output.
- Inputs may be treated as clean text; the catalog, prices, and category tree are
  static during the event; sessions are isolated and do not require concurrency
  stress testing.

## Provided Kit and Permitted Resources

The participant kit supplies the frozen catalog, a weak Python BM25 starter, the
Agent interface, a deterministic evaluator, evaluation configuration, baseline
results, data/submission documentation, and a catalog SHA256 checksum. Participants
may replace the starter with rule-based, keyword, dense, hybrid, reranking, local
model, or legally accessible external-API methods. The organizer supplies no model
credits, keys, tokens, or hosted inference.

Original references from the brief:

- [Participant repository](https://github.com/TechJam2026/techjam-conversational-search)
- [Participant-kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit)
- [Amazon Reviews 2023 documentation](https://amazon-reviews-2023.github.io/)

The frozen competition files should be used directly; reconstructing the full
upstream Amazon dataset is unnecessary.

## Brief-to-Code Status

| Brief requirement | Current status | Code/evidence and decision |
|---|---|---|
| Buying/Browsing intent detection | Implemented and active diagnostically | `mercury/intent.py` emits Buying, Browsing, or Mixed decisions; typed plans live in `mercury/planning.py`. The selected release does not enable the score-regressing routed behavior. |
| Multi-route hybrid retrieval | Implemented, independently gated | Sparse broad and category-scoped routes are selected. Conditional routed/dense retrieval exists behind configuration; the measured routed candidate did not pass promotion, and dense routed assets were not evaluated in that cycle. |
| Semantic ranking | Implemented and selected | A pinned local MiniLM cross-encoder reranks 30 candidates. This is neural semantic ranking, not a generative LLM. Optional LLM use remains unjustified and disabled. |
| Information accumulation and intent override | Implemented and selected | The source-linked preference ledger retracts stale assertions, preserves unrelated constraints, records polarity/hardness/provenance, and handles no-preference replies. |
| Proactive clarification | Partly implemented, gated candidate rejected | Named attributes are single-use; generic prompts vary and stop after an unproductive reply. Intent-aware question selection exists but did not pass promotion. |
| Immediate retrieval cutoff for over-generality | Implemented and gated; candidate rejected | A target-independent pre-expensive-retrieval gate can use a sparse minimal probe or clarify before catalog access. The minimal-probe unseen-development arm reduced score/MTTC and remains disabled. The older cutoff still only reduces post-retrieval reranking. |
| Runtime context adaptation | Implemented as gated deterministic orchestration | Profile priors, inferred-soft decay, route changes, and policy changes are configurable. The combined adaptation candidate did not pass promotion; explicit session intent remains authoritative. |
| Dynamic truncation | Implemented in gated retrieval/rerank controls | Candidate and rerank budgets are bounded and configurable. Dynamic rerank expansion beyond the selected 30-prefix is not selected. |
| Unknown-safe price handling | Implemented and selected | Price is a bounded soft ranking preference. Missing, malformed, lower-bound, or inconsistent evidence is neutral and never excludes a product. |
| Metrics and reproducibility | Implemented | The unchanged evaluator, comparison suite, stage diagnostics, private-like capabilities, model pinning, checksums, and fallback checks are documented in the repository. |

Do not re-add rows marked implemented as generic TODOs. Improve them only through
a named, gated hypothesis with new evidence. Keep gated code and configuration
when it supports reproducibility, ablation, or a documented negative result.

## Selected Release and Experiment Boundary

The persistent release decision is:

- **Submission/reliability:** `configs/selected.json`, using the 30-candidate
  MiniLM rerank prefix.
- **Public-score demonstration only:** D120 measured `0.807170` on the already-used
  public set but failed fresh screening and latency criteria. It is not selected.
- **Potential future candidate:** D60 may be considered only after a registered
  evaluation on new unseen sessions.

The selected configuration keeps grouped alternatives, neural weight `0.75`, a
four-question bounded `other` policy, rerank depth 30, and soft-price weight
`0.02`. Behavior-changing experiments stay disabled unless they pass the
promotion protocol in [the plan](../plan.md). Historical scores remain attached
to the exact source and configuration that produced them.

## Deliverables and Current Ownership Boundary

The supplied brief requires:

1. A Devpost description covering the problem, solution, development tools,
   APIs, libraries/frameworks, datasets, and assets.
2. A public repository with structured/commented code, overview, setup,
   reproduction, limitations/future work, and team contributions when applicable.
3. A public three-minute YouTube demo linked from Devpost. For this backend track, an API,
   inference, or results walkthrough is acceptable; third-party rights still
   apply.

Repository drafts, setup, report, result evidence, and a real terminal replay
exist. Public repository approval, team details, final video export/upload, video
URL, and Devpost submission remain owner-controlled external actions. Do not mark
them complete from local files alone. Track them in
[the release checklist](RELEASE_CHECKLIST.md) rather than duplicating them in the
model-improvement backlog.

## Documentation Routing for Future AI Work

- Use this file for challenge intent, rubric framing, constraints, and current
  requirement-to-code status.
- Use [the plan](../plan.md) for active model/product refinement and promotion
  gates.
- Use [pipeline evolution results](PIPELINE_EVOLUTION_RESULTS.md) and
  [merge decisions](MERGE_DECISIONS.md) for completed and rejected experiments.
- Use [the report](../REPORT.md) and [scoring guide](SCORING_AND_JUDGING.md) for
  claims and measured evidence.
- Use [the release checklist](RELEASE_CHECKLIST.md) for external submission work.
- Update this status table when a feature is promoted, rejected, or removed; do
  not create a second competing rubric summary.
