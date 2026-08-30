# Mercury Pipeline Evolution Plan

## Purpose

Evolve Mercury from one shared conversational retrieval pipeline into an explicit intent-routed hybrid system while preserving reversible session state, conservative handling of missing metadata, offline operation, legal Top-10 output, and reproducible evaluation.

This plan is written for incremental execution by Codex. Implement one phase at a time, run the required checks, record the result, and do not enable a component in `configs/selected.json` until it has passed its acceptance gate.

Read [the challenge/rubric alignment](docs/CHALLENGE_RUBRIC_CONTEXT.md) before
changing this plan or runtime behavior. Phases 0-7 below have already been
implemented and evaluated; their descriptions remain because they define durable
feature boundaries, configurations, and acceptance criteria. Phase 8 was
deliberately not activated. Phases 9-14 are now implemented as gated candidates;
none passed promotion on the frozen unseen-target development split. See
[the roadmap results](docs/ROADMAP_IMPLEMENTATION_RESULTS.md). Completed or
rejected items must not be reintroduced as unmeasured TODOs.

## Current Reference Point

The current selected pipeline is:

```text
message
-> deterministic preference ledger
-> classify buying / browsing / mixed intent
-> build a typed retrieval plan from live state
-> query from active positive preferences
-> sparse BM25 retrieval
-> optional category-scoped sparse retrieval
-> reciprocal-rank fusion
-> unknown-safe contradiction guard
-> MiniLM cross-encoder reranking of 30 candidates
-> contradiction guard again
-> bounded soft-price preference (unknown remains neutral)
-> bounded non-repeating question policy
-> fixed Top-10 slate
```

Recorded whole-public result:

| Metric | Selected Mercury |
|---|---:|
| HitRate@10 | 0.895000 |
| MRR | 0.613746 |
| MTTC | 3.245000 |
| TechnicalScore | 0.786724 |

Scenario-level evidence indicates that an explicit router is worth investigating:

| Scenario | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Buying | 0.875000 | 0.562515 | 2.875000 |
| Browsing | 0.912500 | 0.611607 | 2.912500 |
| Intent override | 0.866667 | 0.731944 | 4.600000 |
| Boundary | 1.000000 | 0.686111 | 4.800000 |

These are public development metrics, not estimates of private performance. The existing public reserve has already been consumed and must not be described as fresh holdout evidence.

## Design Principles

1. Deterministic code owns explicit facts, hard constraints, state mutation, and safety checks.
2. Retrieval models own candidate recall; reranking models own semantic ordering among valid candidates.
3. A neural model may rank evidence, but it may not override an explicit rejection, correction, size, object type, or other reliable hard constraint. Catalog price remains soft because its evidence is incomplete and inconsistent.
4. Missing catalog metadata means unknown, not contradicted.
5. LLM use is optional, confidence-gated, schema-validated, and disabled by default until an ablation proves value.
6. Every new behavior must be configurable and independently ablatable.
7. Preserve `configs/selected.json` as the frozen reference until a replacement passes the complete evaluation gate.
8. Never add target IDs, sample IDs, scenario labels, simulator behavior, or public-session-specific rules to runtime code.
9. Continue returning a full legal slate while asking a question unless an official contract requires abstention. The evaluator permits recommendations and a question on the same turn.

## Target Architecture

```text
User message
    |
    v
Deterministic parsing + confidence and unresolved spans
    |
    v
Source-aware state ledger
    |
    v
Intent/specificity router
    |-------------------------|
    v                         v
Buying plan               Browsing plan
precise sparse routes     dense semantic routes
category/object scope     broad sparse recovery
hard constraints          category hypotheses
    |                         |
    |-----------+-------------|
                v
      Coverage-first route fusion
                |
                v
  Deterministic constraint/object guard
                |
                v
      Structured neural reranking
                |
                v
  Deterministic question/slate policy
                |
                v
       Top-10 + clarification
```

The system should support a third `mixed` mode when a product type is known but the request still contains a broad use case or exploratory intent.

## Mechanism Boundaries

| Responsibility | Default mechanism | Optional learned mechanism |
|---|---|---|
| Normalization and explicit negation | Deterministic | None |
| Size, budget, color, brand and category slots | Deterministic | Confidence classifier only |
| Corrections, retractions and no-preference replies | Deterministic ledger | LLM fallback for unresolved language |
| Buying/browsing/mixed routing | Explainable feature score | Small classifier if rules are inconclusive |
| Buying retrieval | Category/object BM25 and reliable structured filters | Dense recovery route |
| Browsing retrieval | Dense retrieval, broad BM25 and query variants | LLM-generated semantic query variants |
| Route fusion | Weighted reciprocal-rank fusion | Learned weights only with proper validation |
| Hard constraints and object/accessory compatibility | Deterministic three-state evidence | Classifier may add evidence, never final authority |
| Semantic ranking | Cross-encoder | Optional local listwise LLM experiment |
| Question selection | Deterministic expected-value policy | None initially |
| Question wording | Fixed templates | Optional wording-only generation |
| Short-term memory | Deterministic ledger | None |
| User profile | Weak prior | Optional embedding/profile distillation |

## Phase 0: Protect the Reference and Extend Measurement

### Work

- Keep the existing selected configuration and recorded result unchanged.
- Add candidate configurations rather than editing `configs/selected.json` during development.
- Extend diagnostics so every turn records:
  - inferred mode, specificity, confidence and routing reasons;
  - active hard, soft, negative and neutral preferences;
  - route names, weights, candidate counts and target-independent overlap;
  - candidate count before and after each guard/ranker;
  - chosen question and the reason it was selected;
  - latency and fallback status by component.
- Extend the evaluation report with fixed-turn retrieval depths where possible: Recall@30, Recall@60 and Recall@120 for each route and the fused union.
- Add a deterministic fixture set for router, parser, object/accessory, scope, negative-feedback and question-policy cases.

### Likely files

- `mercury/agent.py`
- `experiments/run.py`
- `experiments/evaluate_suite.py`
- `experiments/private_like_validate.py`
- `tests/`

### Acceptance gate

- Existing selected output and score remain reproducible.
- Existing tests and lint pass.
- New diagnostics contain no ground-truth or evaluator-only information at inference time.
- Evaluation can attribute a failure to state, retrieval, ranking, policy, or runtime fallback.

## Phase 1: Introduce a Typed Intent Router

### Work

- Add an immutable `IntentDecision` type with:

```python
mode: Literal["buying", "browsing", "mixed"]
specificity: float
confidence: float
hard_constraint_count: int
over_general: bool
reasons: tuple[str, ...]
```

- Implement deterministic routing features:
  - explicit object/category;
  - number of active constraints;
  - hard-language markers such as `must`, `need`, `under`, and explicit rejection;
  - browsing markers such as `ideas`, `exploring`, `something for`, gifts and occasions;
  - use-case language without a product type;
  - unresolved open-vocabulary residual text;
  - whether the current turn overrides earlier intent.
- Route from the distilled live state plus the current message. Do not route from stale raw history.
- Make thresholds configurable.
- Expose reasons in diagnostics.
- Start with rules. Do not introduce an LLM or trained classifier in this phase.

### Likely files

- New `mercury/intent.py`
- `mercury/types.py`
- `mercury/config.py`
- `mercury/agent.py`
- New `tests/test_intent.py`

### Acceptance gate

- Hand-authored buying, browsing, mixed and override fixtures pass.
- Rephrasing tests produce stable routes.
- No route reads evaluator scenario labels.
- Router failure falls back to `mixed`, not to an empty result.

## Phase 2: Replace the Flat Query with a Retrieval Plan

### Work

- Keep `SessionState` as the source of truth, but stop passing only one flattened positive string through the entire pipeline.
- Add a typed `RetrievalPlan` containing:

```python
mode
object_types
category_terms
positive_terms
negative_terms
hard_constraints
soft_preferences
use_case
scoped_features
semantic_queries
```

- Build the plan deterministically from active preferences and the router decision.
- Preserve polarity, hardness, source turn, owner/component scope and alternative groups.
- Keep a lexical query for BM25, a semantic query for dense retrieval, and a structured context for reranking.
- Ensure withdrawn and neutral preferences appear in none of the active search contexts.
- Include negative preferences in reranking context even when they are deliberately excluded from positive BM25 terms.

### Likely files

- `mercury/types.py`
- `mercury/state.py`
- New `mercury/planning.py`
- `mercury/agent.py`
- `tests/test_state.py`
- New `tests/test_planning.py`

### Acceptance gate

- Existing corrections and no-preference tests still pass.
- Structured plans correctly represent grouped alternatives and component ownership.
- The same active ledger always produces the same plan.
- No raw conversational scaffolding leaks into lexical queries.

## Phase 3: Implement Dual-Track Retrieval

### Buying track

- Make category/object-scoped BM25 the dominant route.
- Retain a broad sparse route for metadata mismatch recovery.
- Use dense retrieval as a lower-weight recovery route, not the primary route.
- Apply true filtering only for reliable structured facts. Treat incomplete price, size or material metadata as unknown.
- Use dynamic candidate limits based on specificity and route agreement.

### Browsing track

- Make dense retrieval the dominant semantic route.
- Retain broad sparse retrieval as lexical recovery.
- Search the original use case, distilled scenario and bounded category hypotheses as separate routes.
- Add controlled category/product-type diversity before reranking so one lexical family cannot consume the candidate budget.

### Mixed track

- Use the buying product scope with browsing semantic expansion.
- Fuse both route families with balanced weights.

### Implementation notes

- Reuse `DenseIndex` and `fuse_routes` initially.
- Keep model loading local-only and retain sparse fallback behavior.
- Do not enable dense retrieval globally based on the earlier negative hybrid result. The new experiment is conditional routing, not a repetition of global dense fusion.
- Prefer route-specific config blocks or clearly named flat configuration fields.

### Likely files

- `mercury/retrieval.py`
- `mercury/neural.py`
- `mercury/agent.py`
- `mercury/config.py`
- New candidate configs under `configs/`
- `tests/test_retrieval.py`
- `tests/test_agent.py`

### Acceptance gate

- Fused Recall@120 does not regress by more than 0.02 overall or in buying.
- Browsing Recall@30 or Recall@60 improves on independently authored cases.
- Missing dense assets still produce ten valid unique catalog IDs through sparse fallback.
- Route latency and memory remain within the existing documented engineering caps.

## Phase 4: Add Object, Accessory and Component Guards

### Work

- Derive a normalized product type from category paths and high-confidence title patterns.
- Define a compact, data-driven compatibility map for common catalog families: requested object, compatible accessory, replacement component, and unrelated object.
- Add explicit scope to feature preferences, for example `gold zipper` versus `gold bag`.
- Resolve owner/component relationships using deterministic patterns first.
- Add penalties in this order:
  1. proven hard-constraint contradiction;
  2. requested-object versus accessory mismatch;
  3. component-scope contradiction;
  4. soft preference mismatch.
- Preserve unknown evidence. A missing product-type classification must not become a contradiction.
- Apply the guard before candidate truncation and again after neural reranking.

### Likely files

- `mercury/catalog.py`
- `mercury/types.py`
- `mercury/state.py`
- `mercury/ranking.py`
- New `mercury/product_types.py`
- `tests/test_catalog.py`
- `tests/test_ranking.py`
- Private-like capability fixtures

### Acceptance gate

- Accessory-above-requested-object fixture failures reach zero.
- Body/component scope fixtures pass without excluding unknown products.
- Explicit negatives cannot return to the top group after reranking.
- No special rules reference public target IDs or titles.

## Phase 5: Give the Cross-Encoder Structured Context

### Work

- Replace the flat query/document pair with stable labeled text.

Example query context:

```text
Mode: buying
Object: handbag
Must have: canvas; zipper
Must not have: leather
Preferred use: daily commuting
Budget: at most 80
```

Example product context:

```text
Product type: handbag
Title: ...
Categories: ...
Features: ...
Material: ...
Price: ...
```

- Keep deterministic truncation priorities so object type, hard positives and negatives are never removed before low-value description text.
- Retain the post-rerank deterministic guard.
- Compare score blending by rank and by normalized model score only if normalization is fitted without using the final evaluation rows.
- First benchmark the existing MiniLM model with structured context.
- Only after that, benchmark one stronger local cross-encoder under the same candidate sets, latency limits and model-asset verification.

### Likely files

- `mercury/neural.py`
- `mercury/model_assets.py`
- `experiments/prepare_models.py`
- `tests/test_neural.py`
- Candidate configs under `configs/`

### Acceptance gate

- MRR improves without a HitRate@10 regression greater than 0.02.
- Negative, accessory and scope cases remain correct after neural ranking.
- Doubling rerank work is rejected unless the gain passes the repository's practical threshold.
- Missing or corrupt model assets still use the legal sparse fallback.

## Phase 6: Make Clarification Intent-Aware

### Work

- Detect over-generality from candidate count, route disagreement, top-score gap, known-constraint count and facet uncertainty.
- Buying questions should target the missing attribute most likely to divide the leading valid candidates.
- Browsing questions should first seek product type, use case or the most decision-relevant style dimension.
- Do not ask an attribute that the user already answered with no preference.
- Avoid repeating an unproductive question.
- Preserve `other` as a bounded fallback because it is useful in the official simulator, but do not make it the unconditional first choice for four turns.
- Return the current Top-10 alongside the question. Interpret an immediate retrieval cutoff as avoiding unnecessary expensive reranking, not automatically returning an empty slate.
- Optimize question choice against expected recovery and reciprocal-rank gain, with an explicit turn-cost term.

### Likely files

- `mercury/policy.py`
- `mercury/config.py`
- `mercury/agent.py`
- `tests/test_policy.py`
- Candidate configs under `configs/`

### Acceptance gate

- MTTC does not regress overall or in buying/browsing.
- Boundary and no-preference fixtures do not loop.
- Question-policy ablation beats or matches the current bounded `other` policy on more than one evaluation slice.
- The final turn never asks a question.

## Phase 7: Make Runtime Adaptation Conservative

### Work

- Treat explicit current-session preferences as authoritative.
- Add decay only for inferred soft preferences and unconfirmed category hypotheses.
- Never decay explicit hard constraints or negative feedback.
- Use profile tags and summaries only as weak tie-breaking priors after current-session relevance.
- Record whether each signal came from the current turn, earlier session state, inference, or profile.
- Let routing diagnostics adjust route weights, candidate budgets and question strategy at runtime.
- Do not implement self-modifying code or unconstrained prompt rewriting. Here, dynamic context programming means deterministic re-orchestration from current state and uncertainty.

### Likely files

- `mercury/state.py`
- `mercury/planning.py`
- `mercury/intent.py`
- `mercury/agent.py`
- `tests/test_state.py`
- `tests/test_agent.py`

### Acceptance gate

- Current explicit intent always outranks profile priors.
- Intent override removes incompatible inferred state immediately.
- Replaying the same conversation and profile is deterministic.
- Profile-free and malformed-profile inputs remain legal.

## Phase 8: Optional LLM Fallback Experiment

Do not start this phase until Phases 1-7 are measured. The current task does not require a paid or hosted LLM, and a cross-encoder is better suited to routine reranking.

### Allowed uses

- Resolve an unresolved span when deterministic parsing confidence is low.
- Produce at most two additional semantic queries for a genuinely broad browsing request.
- Resolve an ambiguous feature-to-component relationship.
- Experiment with listwise ordering of a very small, already valid candidate prefix.

### Prohibited uses

- Directly mutate session state without schema validation.
- Override hard constraints or the deterministic guard.
- Invent catalog facts, IDs, prices or availability.
- Become mandatory for legal output.
- Send user profiles, catalog data or conversation history to a hosted service without explicit configuration and documentation.
- Run on every turn merely to satisfy the phrase `LLM semantic ranking`.

### Interface requirements

- Define a provider-neutral interface with timeout, token and latency limits.
- Require strict JSON output and validate every field.
- Record confidence and provenance.
- Fall back to the deterministic plan on parse failure, timeout or unavailable credentials.
- Keep the feature disabled in default and selected configs until it produces a reproducible practical gain.

### Acceptance gate

- Improvement exceeds the predeclared practical threshold on independent fixtures and cross-validation.
- No hard-constraint, privacy, offline-fallback or legality regression.
- Cost, token usage and latency are reported.
- The README accurately distinguishes neural cross-encoder ranking from generative LLM use.

## Active Refinement Roadmap: Robustness Under a Fixed Cost Envelope

The next objective is not to add every available model or retrieval route. It is
to improve unseen-session recovery and dialogue decisions while keeping worst-case
work explicit and bounded. Each phase below must have its own candidate config,
registered hypothesis, maximum per-turn/session budget, and failure rollback. The
selected 30-candidate release remains unchanged until a candidate passes the full
promotion rule.

### Phase 9: Establish New Unseen Evidence Before More Tuning (Implemented)

#### Work

- Create a target-disjoint and user-disjoint evaluation pack that is not derived
  from the 200 public labels. Prefer organizer-approved new sessions; otherwise use
  independently authored conversations against catalog products selected before
  writing the dialogue.
- Cover paraphrases, intent overrides, explicit negative feedback, ambiguous
  product types, accessories versus primary products, missing metadata, conflicting
  price evidence, no-preference replies, and vague browsing starts.
- Freeze session construction rules, target IDs, prompts, evaluator version, and
  random seeds before running a new candidate. Keep a sealed final fold that is not
  used to set thresholds.
- Add grouped cross-validation by target and user for development. Report mean,
  worst fold, bootstrap interval, scenario slices, and regression counts rather
  than choosing from a single aggregate.
- Add a failure taxonomy that separates state, intent, retrieval recall, rerank
  ordering, constraint evidence, question choice, and runtime fallback.

#### Cost control

This is evaluation work, not runtime work. Reuse cached sparse indexes, candidate
sets, and deterministic model outputs where the candidate/config inputs are
identical. Set a fixed experiment budget before seeing results and stop candidates
that fail correctness or early-fold gates.

#### Acceptance gate

- No target or user crosses train/tuning/final groups.
- Session authors cannot inspect Mercury rankings while writing expected intent.
- The pack detects at least one known failure of the selected system and contains
  no runtime target lookup or mock-ASIN dependency.
- No future D60, D120, routing, or dialogue promotion relies only on the consumed
  public set.

### Phase 10: Add a Pre-Expensive-Retrieval Sufficiency Gate (Implemented, Not Promoted)

#### Work

- Add a cheap deterministic `RetrievalSufficiencyDecision` immediately after state
  update and typed plan construction, before catalog retrieval and neural reranking.
- Base it only on information available at inference: known object/category,
  active hard/soft/negative signals, unresolved residual language, intent
  confidence, specificity, prior answer productivity, turn number, and whether a
  previous legal slate is cached. Do not use target-aware statistics.
- Produce one of three explicit actions:
  - `retrieve`: run the normal selected pipeline;
  - `minimal_probe`: run one bounded sparse route with no neural rerank, return a
    legal slate, and ask the highest-value clarification;
  - `clarify_first`: ask a question before any catalog search. Keep this as a
    separately gated experiment because an empty first-turn slate incurs a scored
    miss and may worsen MTTC.
- Make `minimal_probe` the first practical candidate. It implements the spirit of
  immediate cutoff by avoiding broad multi-route and neural work while preserving
  recommendations. Compare strict `clarify_first` only if the evaluator contract
  and unseen evidence justify the turn cost.
- Permit at most one consecutive deferred/cheap-probe turn. Always retrieve on the
  final turn, after a productive answer, or when no eligible non-repeating question
  remains. On errors, fall back to the selected retrieval path.
- Record decision reasons, avoided routes, avoided rerank pairs, returned slate
  size, latency, and whether the next answer reduced uncertainty.

#### Persistent configuration

Keep the behavior disabled by default and expose bounded fields such as:

```text
retrieval_sufficiency_gate = false
insufficient_action = "minimal_probe"  # or "clarify_first"
max_deferred_turns = 1
minimal_probe_limit = 30
minimum_retrieval_specificity = ...
```

Names may change during implementation, but the independent gate, hard maximum,
and selected-config default must remain.

#### Acceptance gate

- The selected config remains byte-for-byte unchanged.
- The gate never loops, suppresses a productive override, asks on turn 10, or
  returns an invalid/duplicate ID.
- On new unseen sessions, MTTC and HitRate stay within promotion limits while p95
  rerank pairs or p95 latency decreases materially.
- Report strict clarification's empty-slate rate separately; do not hide its
  first-turn misses inside averages.

### Phase 11: Use an Uncertainty-Triggered Compute Cascade (Implemented, Not Promoted)

#### Work

- Keep 30 reranked candidates as the normal path. Escalate to at most D60 only
  when target-independent diagnostics indicate a likely ranking/coverage failure,
  such as low route overlap, flat cross-encoder margins, excessive guard
  displacement, unstable Top-10 membership across cheap routes, or unresolved
  object hypotheses.
- Define one scalar escalation score from calibrated diagnostics and freeze its
  threshold using grouped development folds. Avoid a growing tree of special-case
  thresholds.
- Cap escalation to one per turn and a small fixed number per session. Never chain
  30 to 60 to 120 or run multiple rerankers as an ensemble.
- Reuse D30 scores when expanding to D60 so the additional work is only the next
  30 pairs. Cache by normalized plan, state revision, document mode, and model
  revision.
- Keep D120 as historical public demonstration evidence only.

#### Acceptance gate

- Worst-case neural work is D60, so cost grows by at most 2x on escalated turns
  and remains linear in a fixed cap rather than combinations of routes/models.
- The fraction of escalated turns and per-session p50/p95 compute are reported.
- D60 promotion evidence comes from the new unseen protocol and passes the
  registered latency, memory, score, and regression gates.
- If uncertainty does not predict failures better than a fixed D30 baseline, keep
  D30 and reject the cascade.

### Phase 12: Improve Intent Representation with Bounded Multi-Hypothesis Retrieval (Implemented, Not Promoted)

#### Work

- Represent ambiguous object/use-case interpretation as at most two typed
  hypotheses rather than flattening all residual words into one query.
- Generate hypotheses deterministically from the live ledger, catalog taxonomy,
  and bounded synonym/category maps. An optional model may suggest hypotheses only
  behind Phase 8's schema, timeout, and fallback rules.
- Allocate one fixed candidate budget across hypotheses and routes; hypotheses do
  not each receive a full budget. Use coverage-aware fusion so the dominant lexical
  reading cannot consume every admitted candidate.
- Improve negative-feedback handling by distinguishing “not this item,” “not this
  product type,” and “not this attribute value.” Apply the narrowest supported
  retraction/exclusion and preserve unrelated intent.
- Store hypothesis provenance and retire a contradicted hypothesis immediately
  after an override or answer.

#### Acceptance gate

- At most two hypotheses and a fixed total retrieval/candidate budget are enforced
  in configuration and tests.
- Paraphrase and ambiguity slices improve fused Recall@30 or Recall@60 on unseen
  targets without a buying HitRate regression beyond the promotion limit.
- Explicit negative feedback never broadens into an unsupported category-wide
  exclusion.
- Missing taxonomy or hypothesis failure falls back to the selected lexical plan.

### Phase 13: Ask for Information, Not Merely a Different Sentence (Implemented, Not Promoted)

#### Work

- Track a structured `question_goal` in addition to `ask_attribute` and visible
  wording. Two differently worded questions seeking the same unresolved fact count
  as a repeat.
- Mark each answer as productive, neutral/no-preference, contradictory, or
  unresolved. Do not ask the same goal again after any terminal outcome.
- Estimate question value from a bounded minimal probe: expected partition balance,
  Top-10 instability, unresolved hard-constraint value, and turn cost. Do not scan
  the entire catalog separately for every possible question.
- Couple the question to the sufficiency action: ask only when the expected value
  of the answer exceeds the cost of delaying or reducing retrieval. Otherwise
  retrieve and return the best current slate without a question.
- Prefer questions that distinguish current intent hypotheses or remove likely
  failure modes. Do not ask a generic “anything else?” merely to consume the
  allowed `other` count.

#### Acceptance gate

- Exact and semantic question-goal repeat rates are zero in authored multi-turn
  fixtures.
- Boundary/no-preference sessions terminate questioning cleanly, and turn 10 never
  asks.
- On unseen sessions, question productivity rises without worsening MTTC or target
  recovery beyond the promotion limits.
- Strict one-`other` remains a recorded rejected experiment; do not silently
  reproduce it under a new policy name.

### Phase 14: Calibrate or Compress One Model at a Time (Tooling Implemented)

#### Work

- Calibrate existing cross-encoder margins on grouped development folds before
  adding a larger model. Use calibration to drive the Phase 11 escalation decision,
  not to invent product evidence.
- Benchmark only one stronger or distilled local cross-encoder candidate at a
  time on frozen candidate sets. Compare conditional MRR, latency, RSS, asset size,
  and fallback behavior.
- Prefer quantization or teacher-to-small-model distillation performed offline if
  it preserves ranking quality. Full-parameter foundational-model training remains
  out of scope.
- Reject listwise ensembles and per-route model duplication unless they fit the
  same fixed pair budget and demonstrate a practical unseen gain.

#### Acceptance gate

- The candidate is optional and local, with pinned assets and a legal sparse
  fallback.
- Total model calls/pairs are capped independently of route or hypothesis count.
- Calibration is fitted without the final unseen fold and is reported with
  reliability error as well as downstream score.
- A larger asset is promoted only if its unseen gain clears the practical threshold
  and its feasibility costs remain defensible under the rubric.

### Phase 15: Reduce Handwritten Heuristics (Planned)

This phase targets the remaining vocabulary, mapping, weight, and threshold
choices that are likely to generalize poorly. Learned components may propose or
rank interpretations, but the deterministic authority in the Design Principles
remains unchanged. In particular, explicit corrections, negation, budget and size,
the hard/soft ledger, soft price handling, missing-as-unknown behavior, catalog-ID
legality, question non-repetition, and runtime caps stay deterministic.

#### Priority refactors

| Area | Problem | Better approach |
|---|---|---|
| `mercury/intent.py` | Two regex lists, handwritten weights, and fixed thresholds can give conflicting signals for phrases such as “I need ideas” and miss unseen paraphrases. | Combine structural state features with a small semantic classifier calibrated on independently authored paraphrases. Return `mixed` or `uncertain` at low confidence. |
| `mercury/hypotheses.py` | Fixed use-case mappings such as `wedding → dresses/jewelry` and `work → shirts/bags` are narrow and potentially biased. | Derive use-case/category associations from catalog text or frozen offline retrieval statistics. Require adequate support and confidence, and retain the two-hypothesis shared budget from Phase 12. |
| `mercury/state.py` | Manually maintained dictionaries for categories, materials, styles, use cases, and features lose coverage on unseen products and language. | Generate versioned canonical values and aliases from the catalog, then apply semantic extraction only to unresolved spans. |
| `mercury/product_types.py` | Product families, components, accessories, and aliases form a small hand-selected taxonomy. | Build a versioned taxonomy from catalog category paths and reliable metadata, with `unknown` as the safe fallback and deterministic compatibility checks. |
| `mercury/policy.py` | Question values, answer probabilities, attribute order, and special cases are manually weighted. | Estimate bounded question utility from independently authored sessions: answerability × expected candidate reduction × non-repetition, minus turn cost. Preserve Phase 13 eligibility and non-repetition rules. |

Stopwords, token truncation, fusion/ranking weights, and cascade weights are also
heuristic, but are lower priority because the selected BM25-based D30 path is more
reliable than the semantic candidates tested so far. Revisit them only through
registered one-variable ablations after the five refactors above.

#### Calibrated intent model

Use the current message and distilled live state to compute:

- structural features: known object, specified attributes, hard exclusions,
  use-case-only request, unresolved language, and correction/preference change;
- one cached semantic score from a pinned local encoder or lightweight classifier.

Compare the current rules with three progressively richer baselines:

1. Multinomial logistic regression on structural features, replacing the fixed
   buying/browsing weights with inspectable learned coefficients.
2. A frozen local sentence embedding with a linear classifier.
3. A hybrid linear model combining structural and semantic features.

Use a nonlinear model only after a registered failure of the linear baselines. A
hosted LLM is unnecessary, and merely moving existing weights into configuration
does not address generalization.

#### Train-validation-test protocol

- Independently author direct, indirect, mixed, correction, override, vague,
  conflicting, and out-of-vocabulary paraphrases from frozen intent cards without
  inspecting Mercury predictions.
- Label `buying`, `browsing`, and `mixed` from the authoring specification. Compare
  an explicit `uncertain` class with confidence-based abstention on validation only.
- Default to a grouped 70/15/15 train/validation/test split. Group by author/user,
  paraphrase family, intent card, and target/product family so near-duplicates do
  not cross folds.
- Fit coefficients on train; choose regularization, probability calibration, and
  the `uncertain` threshold on validation; open the sealed test once.
- Do not train or tune on the consumed public set or `unseen-v1`. Freeze dataset,
  split, feature, seed, encoder, coefficient, and calibration hashes before test.
- Report macro F1, per-class precision/recall, confusion matrix, log loss, Brier
  score, calibration error, abstention rate, latency, and authored-slice results.

#### Remaining refactor constraints

- Catalog-derived aliases and taxonomies must be versioned against a catalog hash,
  use minimum support/confidence, preserve `unknown`, and never learn from evaluator
  targets or replies.
- Catalog-derived hypotheses retain provenance, at most two hypotheses, and one
  fixed shared candidate budget.
- Semantic extraction only proposes values for unresolved spans and cannot replace
  conflicting explicit ledger evidence.
- Question utility scores only the bounded eligible goals from Phase 13 and reuses
  the Phase 10 minimal probe rather than scanning the catalog for every attribute.
- Runtime work remains bounded to one cached intent encoding per informative turn,
  a frozen lightweight model, D30 by default, and existing D60/session caps.

#### Acceptance gate

The intent model must first beat the rules-only baseline on the sealed intent test
without deterministic fixture or cost regressions. Then freeze it and register one
joint retrieval-and-dialogue candidate on a newly authored target/user-disjoint
session pack. Intent accuracy alone cannot justify release: the candidate must pass
the global TechnicalScore, HitRate, correctness, latency, memory, fallback, and
compute promotion rules before routing changes in `configs/selected.json`.

## Evaluation Protocol

### Stage metrics

Track these in addition to the official end-to-end score:

| Stage | Metrics |
|---|---|
| Intent router | Macro F1, per-class precision/recall, confusion matrix, log loss, Brier score, expected calibration error, abstention and fallback rate |
| State parsing | Slot precision/recall, negation accuracy, retraction accuracy |
| Retrieval | Per-route and fused Recall@30/@60/@120 |
| Ranking | MRR conditional on target retrieval, Top-1 and Top-10 movement |
| Constraints | Top-10 hard-violation rate, unknown-preservation rate |
| Product type | Accessory/object error rate, component-scope error rate |
| Dialogue | MTTC, question answerability, repeated/unproductive question rate |
| Runtime | Cold start, p50/p95 latency, RSS, tokens and fallback turns |

### Completed roadmap sequence and next evidence

The first roadmap cycle ran one primary ablation at a time:

1. Reproduced the selected D30 reference at `0.788784`.
2. Froze the public-target-excluding development/final evidence and failure taxonomy.
3. Rejected the minimal sparse sufficiency probe at `0.760716`.
4. Kept the D30-to-D60 cascade gated at `0.789097`; its `+0.000313` delta did not pass.
5. Rejected two-hypothesis retrieval at `0.768430`.
6. Rejected strict semantic value-gated dialogue at `0.638466`.
7. Calibrated selected-run neural margins, but did not promote a threshold or larger model.

The earlier router, routed retrieval, product guard, structured reranking,
intent-aware policy, and runtime-adaptation experiments are complete. Their
negative or non-promotable results are recorded in
[pipeline evolution results](docs/PIPELINE_EVOLUTION_RESULTS.md); do not rerun or
combine them without a new registered hypothesis and unseen evidence. Phase 8's
LLM fallback is still optional and should not displace the bounded active sequence.

For every candidate, compare against both `configs/selected.json` and `configs/sparse_fallback.json`. Report whole-public results as development evidence only. Use target-disjoint cross-validation or independently authored synthetic/private-like cases for new tuning decisions because no untouched public reserve remains.

### Promotion rule

A candidate may replace the selected pipeline only when it:

- achieves a predeclared practical TechnicalScore improvement, initially `>= 0.01`;
- does not reduce overall or buying HitRate@10 by more than `0.02`;
- does not introduce a correctness regression in the private-like capability pack;
- passes all unit tests, Ruff and `pip check`;
- has zero unexpected fallback turns in the measured neural run;
- preserves legal unique catalog recommendations and turn limits;
- documents latency, memory, model assets, tokens and limitations;
- has a frozen config and source snapshot before final comparison.

## Required Verification Commands

```bash
python -m unittest discover -s tests -q
python -m ruff check .
python -m pip check

python -m experiments.private_like_validate \
  --config configs/CANDIDATE.json \
  --output runs/CANDIDATE-private-like

python -m experiments.evaluate_suite \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output runs/CANDIDATE-suite \
  --candidate CANDIDATE=configs/CANDIDATE.json
```

Use a new output directory for every run. Do not overwrite earlier experiment evidence.

## Suggested Codex Execution Order

Give Codex one phase at a time. For each phase, instruct it to:

1. Read this plan and the affected runtime, tests and experiment code.
2. Check the dirty worktree and preserve unrelated user changes.
3. Add tests before or alongside behavior changes.
4. Keep the capability behind configuration when it can affect ranking or policy.
5. Run focused tests, then the full test and lint suite.
6. Run the private-like pack.
7. Run the public comparison only after correctness checks pass.
8. Summarize changed files, measured deltas, regressions, resource changes and remaining uncertainty.
9. Stop and report a negative experiment instead of enabling a component that does not pass its gate.

Recommended next implementation request:

```text
Implement only Phase 15's intent-dataset and classifier-diagnostics milestone;
do not change runtime routing or configs/selected.json. Freeze a grouped 70/15/15
independently authored intent dataset, then compare the rules-only, structural,
semantic-linear, and hybrid-linear baselines under Phase 15's fitting, calibration,
reporting, and leakage rules. Stop after sealed-test diagnostics; changing routing
requires a separate joint candidate on a new downstream unseen pack.
```

## Definition of Done

The pipeline evolution is complete when Mercury demonstrably:

- distinguishes buying, browsing and mixed intent without evaluator leakage;
- uses high-precision retrieval for buying and broader semantic retrieval for browsing;
- represents positive, negative, hard, soft, alternative and component-scoped intent explicitly;
- prevents accessories and proven contradictions from outranking valid requested objects;
- gives the reranker structured intent while preserving deterministic final authority;
- asks non-repetitive questions chosen for expected search value;
- adapts route and policy behavior from live state without unsafe self-modification;
- remains reproducible, offline-capable and legal when optional models fail;
- improves measured performance under the promotion rule rather than only adding architectural complexity.
