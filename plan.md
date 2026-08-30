# Mercury Private-Robustness and Efficiency Plan

## Purpose

The public development score is now strong enough that further public-set tuning
is more likely to overfit than to improve the organizer-held private result. The
next objective is to improve transfer to unseen targets, categories, metadata
conditions, and dialogue phrasing while reducing local neural work and latency
without a significant accuracy or robustness loss.

Execute one bounded experiment at a time. Keep behavior-changing work behind a
candidate configuration, freeze its hypothesis and evaluation data before
measuring it, and do not modify `configs/selected.json` unless every promotion
gate passes.

Read [the challenge and rubric context](docs/CHALLENGE_RUBRIC_CONTEXT.md) and
[the scoring guide](docs/SCORING_AND_JUDGING.md) before changing runtime behavior.

## Current Evidence

The current guarded-paging D30 release reports:

| Evidence | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Consumed public development set, 200 sessions | 0.970000 | 0.641633 | 2.905000 | 0.839390 |
| Frozen unseen development proxy, 80 sessions | 0.887500 | 0.636781 | 3.300000 | 0.788784 |
| Fresh frontier control, 160 sessions | 0.918750 | 0.618284 | 3.356250 | 0.797735 |
| Fresh page-local control, 160 sessions | 0.925000 | 0.635933 | 3.156250 | 0.810155 |

The public score and 97% public HitRate are strong engineering results, but the
roughly `0.789–0.810` target-disjoint scores are the better warning about private
transfer. None of these synthetic packs is organizer-private evidence.

The selected runtime uses a local MiniLM cross-encoder, reranks at most 30 of 120
retained candidates, and reports no paid API use or completion tokens. The
reported prompt tokens are local model input tokens: reducing them primarily
reduces CPU work and latency rather than a hosted inference bill.

## Completed Work and Archived Decisions

The following foundations are complete and should not remain active implementation
phases:

- reversible source-aware preference state, corrections, alternatives, negation,
  no-preference handling, and explicit override paging reset;
- typed intent and retrieval diagnostics;
- broad sparse retrieval, deterministic guards, D30 MiniLM reranking, soft-price
  handling, early paging, legal unique Top-10 output, and offline fallback;
- private-like regression tooling, target-disjoint preparation, stage diagnostics,
  resource measurement, judge showcase, and failure-path tests.

Historical implementation detail and negative results remain in:

- [pipeline evolution results](docs/PIPELINE_EVOLUTION_RESULTS.md)
- [robustness roadmap results](docs/ROADMAP_IMPLEMENTATION_RESULTS.md)
- [pipeline refinement results](docs/PIPELINE_REFINEMENT_RESULTS.md)
- [frontier reranking protocol](docs/FRONTIER_RERANK_PROTOCOL.md)
- [page-local reranking protocol](docs/PAGE_LOCAL_RERANK_PROTOCOL.md)

Do not repeat the following merely under a new name unless a materially new,
pre-registered hypothesis addresses the recorded failure:

- global seen-aware exclusion;
- D60 or D120 reranking as a general solution;
- structured, lexical-window, or protected document packing;
- multi-hypothesis retrieval;
- strict semantic clarification or retrieval deferral;
- dense retrieval added only for architectural complexity;
- page-local reranking with the already-tested trigger and budget;
- listwise ensembles or hosted LLM calls without a fixed need and budget.

## Non-Negotiable Runtime Invariants

1. Runtime code never reads target IDs, sample IDs, evaluator scenario labels,
   future turns, or simulator outcomes.
2. Deterministic state owns explicit facts, corrections, negation, hard/soft
   status, and provenance.
3. Missing metadata means unknown, not contradicted.
4. Learned models may rank or propose evidence but cannot override explicit
   rejection, correction, object type, size, budget, or another reliable fact.
5. Every response contains only unique, catalog-valid IDs and respects the legal
   Top-10 and turn limits.
6. Optional model failure returns a legal selected-style fallback.
7. Every cache key includes enough model, query, state, document, and catalog
   provenance to prevent stale reuse.
8. Current explicit session intent outranks profile-derived or inferred priors.
9. The selected release remains unchanged until a source-frozen candidate passes
   independent correctness, quality, latency, and cost gates.

## Target Runtime

```text
Message
→ deterministic normalization and preference ledger
→ typed retrieval plan
→ BM25 retrieves up to 120 candidates
→ cheap admission scorer evaluates all 120
→ confident request: MiniLM reranks 20
→ uncertain request: MiniLM reranks 30
→ deterministic contradiction and legality guards
→ selected page counter and bounded question policy
→ Top 10 recommendations
```

The cheap scorer and adaptive depth are later candidates, not authorization to
change the selected runtime immediately.

## Phase 1: Behavior-Preserving Neural Efficiency

Start with optimizations that must preserve recommendation IDs and ordering.

### 1A. Exact pair-logit cache

Add a bounded, thread-safe LRU cache for deterministic cross-encoder logits. A
key must include:

- reranker kind and pinned model revision;
- document serializer version and mode;
- complete rerank query or its collision-resistant digest;
- state revision where relevant;
- product ID and product/document fingerprint;
- maximum sequence length and any setting that can affect the score.

Score only cache misses, restore logits in original candidate order, and expose
hit/miss/eviction diagnostics. Resetting a session must not corrupt other sessions,
and model or document-version changes must invalidate old entries.

### 1B. Identical-document grouping

Within a rerank call, group candidates whose exact serialized model document is
identical. Score one representative and reuse its logit for the others. Preserve
every distinct catalog ID and the existing stable tie order. Do not group merely
similar products.

### 1C. Batch and CPU-thread benchmark

The current D30 path uses inference batches of 16, requiring two model batches.
Benchmark batch sizes `16`, `30`, and `32`, and CPU thread counts `2`, `4`, `6`,
and `8` on the actual submission machine. Hold source, model, candidates, and
queries fixed. Compare exact or tolerance-bounded logits, final ordering, p50,
p95, peak RSS, and cold start.

Prefer configuration over hardware-specific branching. Promote only settings
that are stable within the submission memory limit.

### Phase 1 gate

- Exact response parity with selected across public development, fresh proxy,
  private-like, and metamorphic suites.
- Zero new fallbacks, invalid IDs, or cache-provenance failures.
- At least 15% lower warm p95 or at least 20% fewer actually evaluated pairs on
  workloads with reusable pairs.
- No peak-memory regression beyond the predeclared cap.

### Phase 1 execution status — 30 August 2026

- **1A exact pair-logit cache:** implemented behind
  `neural_logit_cache`, with an 8,192-entry thread-safe LRU, complete score
  provenance, per-turn/cumulative diagnostics, cross-session reuse, and exact
  shopper-visible parity. It reduced a fixed exact-reuse workload from 1,200 to
  30 evaluated pairs, but exceeded the registered memory cap on the unseen
  proxy and regressed ordinary-workload p95. It remains opt-in and unselected.
  See [the cache protocol and result](docs/PAIR_LOGIT_CACHE_PROTOCOL.md).
- **1B identical-document grouping:** rejected at feasibility before runtime
  implementation. Exact grouping could save at most 11/13,710 public pairs and
  2/5,460 unseen pairs, far below the 20% gate. See
  [the feasibility audit](docs/IDENTICAL_DOCUMENT_GROUPING_FEASIBILITY.md).
- **1C batch/thread matrix:** all 12 registered cells preserved ranking and
  stayed within `1e-6` cross-batch logit drift. The fastest p95 cell, four
  threads / batch 30, improved only 6.63% versus the selected four threads /
  batch 16 and failed the 15% gate. Selected settings remain unchanged. See
  [the batch/thread result](docs/NEURAL_BATCH_THREAD_PROTOCOL.md).

Phase 1 is closed without promotion. The next active roadmap boundary is Phase
2 evaluation-matrix construction; do not combine the rejected cache or batch
settings with later learned candidates.

## Phase 2: Stronger Private-Like Evaluation

Do not tune the next learned component on the consumed public set or any target
pack whose outcomes informed its design.

### Frozen evaluation matrix

Create source-independent splits with hashes and a consumption ledger:

- target- and loose-title-family-disjoint recovery sessions;
- entire held-out category or product-family groups;
- rating-popularity bands, including low-popularity products;
- metadata-completeness strata: missing price, short title, sparse features,
  contradictory fields, and near-duplicate documents;
- lexically similar wrong-product and accessory hard negatives;
- independently authored buying, browsing, boundary, and override dialogues;
- unseen attribute words and phrasing families.

Do not let session authors inspect Mercury rankings while writing intent cards or
expected state transitions. Group train, tuning, confirmation, and final rows by
author/user, target family, dialogue template, and paraphrase family.

### Metamorphic robustness suite

For conversations whose meaning should remain stable, generate or author:

- clause reordering and punctuation/case changes;
- irrelevant conversational filler;
- spelling, inflection, and ordinary synonym variants;
- alternate correction and override wording;
- negative and no-preference paraphrases;
- equivalent reordered alternatives;
- added unsupported metadata that should remain unknown.

Assert properties rather than target-specific outputs:

- equivalent conversations produce equivalent active state and candidate
  membership;
- irrelevant text cannot retire a preference or remove a valid candidate;
- explicit correction retires only incompatible facts;
- no-change language never triggers an override;
- missing metadata never becomes a contradiction;
- returned IDs remain unique and legal.

### Phase 2 gate

- No target, title family, author/user, or paraphrase family crosses frozen groups.
- Every generated split and tool source has a recorded hash and seed.
- The suite detects known selected-runtime limitations without using runtime labels.
- Confirmation and final outcomes remain unopened until source/config freeze.

### Phase 2 execution status — 30 August 2026

Phase 2 is complete. The source-independent lock contains 480 training, 160
screening, 80 confirmation, and 80 final rows. Target IDs, loose-title families,
whole category groups, authors/users, dialogue templates, paraphrase families,
and unseen-wording families have zero cross-split overlap. Every source and
generated file is hashed under seed `robustness-matrix-20260830-v1`, and the
consumption ledger enforces screening → confirmation → final opening order.

The authored metamorphic suite keeps assertions target-independent. The selected
release passed legality and candidate-membership checks but exposed four active-
state equivalence limitations, so the suite is discriminating rather than a
collection of already-passing examples. Confirmation and final remain sealed.
See [the matrix protocol](docs/PRIVATE_ROBUSTNESS_MATRIX_PROTOCOL.md).

## Phase 3: Cheap Admission Scorer Over All 120 Candidates

The main robustness hypothesis is that a lightweight target-independent scorer
can improve which candidates enter MiniLM while being cheap enough to evaluate
the full retained pool.

### Candidate features

Start with an inspectable linear or histogram-based model using only ordinary
runtime evidence:

- BM25 score, rank, and route agreement;
- normalized title and category overlap;
- exact and partial product-type agreement;
- positive and negative preference coverage;
- proven hard-constraint contradictions;
- catalog-field coverage and metadata completeness;
- bounded price compatibility where price is known;
- query-term coverage by title, category, feature, detail, and description fields;
- optional stable profile prior already allowed by runtime rules.

Do not use target IDs, evaluator labels, popularity derived from test outcomes, or
session fingerprints.

### Training evidence

Generate catalog-derived pseudo-queries from titles, categories, features, and
ordinary attribute combinations. Mine hard negatives from BM25 neighbors and
same-category products. Optionally distill the existing MiniLM on frozen
query-product pairs, but keep training, validation, and final target/title families
disjoint.

Compare, in order:

1. deterministic feature fusion;
2. regularized logistic or pairwise linear scoring;
3. a small tree/histogram model only if linear scoring demonstrably underfits.

Pin feature definitions, preprocessing, seed, coefficients/model bytes, catalog
hash, and training-data hashes. Report admission Recall@20/@30, conditional MRR,
calibration, latency, RSS, and end-to-end metrics.

### Phase 3 gate

- Recall@20 approaches or exceeds selected BM25 admission Recall@30 on more than
  one fresh split.
- Recall@30 does not regress overall or on buying, override, sparse-metadata, and
  held-out-category slices.
- Full-pool scoring adds no more than 10% to selected warm p95.
- Private-like correctness and deterministic guard tests remain clean.
- The model remains optional, pinned, local, and failure-safe.

### Phase 3 execution status — 30 August 2026

Phase 3 was implemented and measured, then rejected. On group-held-out training
proxy evidence, the frozen regularized linear scorer improved admission
Recall@20 from `0.829653` to `0.880126` and Recall@30 from `0.876972` to
`0.914826`; deterministic fusion also improved both. On the first frozen
screening opening, fusion improved TechnicalScore from `0.798906` to `0.804234`
with no scenario HitRate loss, and linear improved it to `0.819922` but lost one
browsing hit.

Neither candidate met the runtime gate: p95 rose from `0.356s` to `0.586s` for
fusion and `0.590s` for linear, increases of roughly 65% against a 10% cap. The
candidate is optional, pinned, local, catalog-hash-bound, and failure-safe, but
it is not promoted. Confirmation remains sealed and `configs/selected.json` is
unchanged. See [the admission protocol](docs/ADMISSION_SCORER_PROTOCOL.md).

### Phase 4 dependency decision — 30 August 2026

Phase 4 is skipped. Its prerequisite is a calibrated, source-frozen Phase 3
admission scorer that passes correctness and resource gates. Phase 3 did not
pass, so opening confirmation or tuning a D20/D30 threshold would violate the
registered dependency. Phase 5 remains independent and is the next active
phase.

## Phase 4: Confidence-Gated D20/D30 Reranking

Only begin after Phase 3 produces a calibrated, source-frozen admission scorer.

Use D20 for confidently ordered requests and retain D30 for uncertainty. The
gate should use a small predeclared feature set, such as:

- admission-score separation near ranks 10, 20, and 30;
- route agreement and candidate-set stability;
- known object/category and hard-constraint coverage;
- unresolved-language count;
- metadata completeness;
- calibrated admission confidence.

Use one scalar decision or a simple monotonic rule. Do not build a phrase-specific
threshold tree. Cache scores, cap every turn at D30, and expose depth/reason/pairs
diagnostics.

### Phase 4 gate

- At least 25% fewer MiniLM pairs and 20% lower warm p95 overall.
- No HitRate@10 loss on screening or confirmation.
- Overall TechnicalScore loss no greater than `0.003` on each fresh comparison.
- No TechnicalScore, HitRate, or MRR loss on buying, override, boundary,
  held-out-category, or sparse-metadata safety slices beyond predeclared sampling
  tolerance.
- No candidate may advance based only on the consumed public score.

## Phase 5: Catalog-Derived Vocabulary and Normalization

Improve open-vocabulary private transfer without weakening deterministic state.

- Build versioned canonical values and aliases from catalog category paths,
  titles, and reliable structured fields.
- Use minimum support and confidence thresholds and retain `unknown` as the safe
  fallback.
- Normalize case, punctuation, plural/inflection variants, hyphenation, and
  conservative spelling variants.
- Apply semantic or fuzzy extraction only to unresolved spans.
- Treat catalog-derived values as proposed evidence; they cannot replace a
  conflicting explicit ledger fact.
- Preserve provenance identifying the catalog version and extraction method.
- Expand product/accessory/component taxonomy from category paths rather than
  public failure titles.

### Phase 5 gate

- Better slot recall on sealed unseen-word and held-out-category tests.
- No precision regression on negation, correction, component scope, or
  no-preference tests.
- No increase in unsupported hard exclusions.
- End-to-end fresh TechnicalScore and HitRate remain non-negative against the
  source-matched control.

### Phase 5 execution status — 30 August 2026

Phase 5 was implemented and measured, then left unselected. A catalog-hash-
bound artifact provides 1,399 supported aliases plus a category-path-derived
object/accessory/component taxonomy. Static explicit parsing owns overlaps;
catalog proposals apply only to unresolved spans, stay soft/additive, retain
version/method provenance, and cannot retire conflicting explicit facts. A
48-family frozen word suite improved slot recall from `0.0` to `1.0` at `1.0`
precision, with all deterministic safety tests clean.

On fresh confirmation, aggregate HitRate improved from `0.937500` to `0.950000`
and TechnicalScore from `0.801393` to `0.815514`, with no fallback. Browsing
HitRate improved, and boundary/override were unchanged, but buying HitRate lost
one of 31 sessions (`0.967742` to `0.935484`). The critical-slice gate rejects
that trade despite the aggregate gain. Final remains sealed and
`configs/selected.json` is unchanged. See
[the vocabulary protocol](docs/CATALOG_VOCABULARY_PROTOCOL.md).

## Phase 6: Optional Compression After the Safe Wins

Consider model compression only after caching, grouping, batching, admission, and
adaptive depth are measured.

Allowed candidates include one at a time:

- offline MiniLM distillation to a smaller local cross-encoder;
- dynamic or static quantization with fixed calibration data;
- ONNX or another portable CPU runtime supported by the submission environment;
- a lower maximum sequence length with a fixed field-preserving serializer.

The earlier protected and lexical serializers reduced tokens but lost ranking
quality, so a new truncation candidate needs a materially different design and a
new protocol. Do not combine model, serializer, depth, and runtime changes in one
unattributable arm.

### Phase 6 gate

- At least 25% lower warm p95, token count, or model asset size.
- Overall TechnicalScore loss no greater than `0.003` on both screening and
  confirmation.
- No HitRate or critical-slice regression.
- Pinned assets, offline loading, corruption tests, and legal sparse fallback.

## Evaluation and Promotion Protocol

### Required metrics

| Area | Metrics |
|---|---|
| State | Slot precision/recall, negation, retraction, override/no-change accuracy |
| Retrieval | Recall@20/@30/@60/@120 by category and metadata slice |
| Admission | Recall@20/@30, hard-negative displacement, calibration |
| Ranking | Conditional MRR, Top-1, Top-10 movement |
| Dialogue | HitRate@10, MRR, MTTC, question productivity/repetition |
| Safety | Hard-violation rate, unknown preservation, valid unique IDs |
| Runtime | Cold start, p50/p95/max, neural pairs, local input tokens, RSS, cache rate, fallbacks |

### Experiment order

1. Register the hypothesis, fixed candidate, resource limits, data, and gates.
2. Generate and hash target-/family-disjoint screening, confirmation, and final
   evidence before runtime implementation.
3. Add invariant and failure tests alongside the candidate.
4. Run focused tests, the complete test suite, Ruff, and dependency checks.
5. Run the private-like and metamorphic suites.
6. Freeze source/config, then run source-matched control and candidate on screening.
7. Reject immediately if any correctness, HitRate, fallback, or resource gate fails.
8. Open confirmation only for a passing frozen candidate.
9. Keep final validation sealed until the final release protocol.
10. Run the consumed public set once only as a descriptive regression check after
    selection; it cannot rescue a failed fresh candidate.

### Default promotion gate

A behavior-changing candidate may replace selected only if it:

- has no HitRate@10 loss on screening and confirmation;
- improves TechnicalScore, unless the experiment is explicitly an efficiency
  candidate allowed a maximum `0.003` loss in exchange for the registered cost
  reduction;
- has no material loss on buying, override, boundary, held-out-category, or
  sparse-metadata slices;
- passes all deterministic, private-like, metamorphic, legality, and failure tests;
- has zero unexpected fallback or agent-error turns;
- meets its predeclared p95, neural-pair, token, RSS, and asset caps;
- uses a frozen source/config and reports all negative results honestly.

Behavior-preserving caching, deduplication, and batching work requires exact
response parity rather than the relaxed efficiency threshold.

## Verification Commands

```bash
python -m unittest discover -s tests -q
python -m ruff check .
python -m pip check

python -m experiments.private_like_validate \
  --config configs/CANDIDATE.json \
  --output runs/CANDIDATE-private-like

python -m experiments.evaluate_suite \
  --output runs/CANDIDATE-suite \
  --candidate CANDIDATE=configs/CANDIDATE.json
```

Use a new output directory for every run and never overwrite prior evidence.

## Recommended Execution Order

1. Implement Phase 1A exact pair-logit caching and prove response parity.
2. Add Phase 1B exact-document grouping under the same parity gate.
3. Run Phase 1C batch/thread benchmarks without changing ranking behavior.
4. Freeze the Phase 2 private-robustness and metamorphic evaluation matrix.
5. Implement and evaluate the Phase 3 lightweight all-120 admission scorer.
6. If Phase 3 passes, evaluate the Phase 4 D20/D30 confidence gate.
7. Independently evaluate Phase 5 catalog-derived normalization.
8. Attempt Phase 6 compression only if the earlier changes do not meet the
   resource objective.

## Definition of Done

This roadmap is complete when Mercury:

- demonstrates stable performance on multiple target-, title-family-, category-,
  author-, and paraphrase-disjoint evaluations;
- handles unseen wording, sparse metadata, corrections, negation, and no-change
  language without evaluator-specific rules;
- improves admission of relevant candidates before neural reranking;
- reduces actual MiniLM pairs by at least 25% and warm p95 by at least 20%;
- loses no HitRate and no more than `0.003` TechnicalScore on fresh confirmation;
- preserves deterministic authority, unknown-safe evidence, offline operation,
  legal output, and explicit failure fallback;
- reports public development performance honestly without treating it as a
  private-score forecast.
