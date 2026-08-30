# Evidence-driven conversational search

## Product

The agent turns a changing shopping conversation into a ranked list of real catalog IDs. It retains supported requirements, retracts corrections, searches broadly, and asks one useful follow-up while showing recommendations. It never purchases anything or invents product specifications.

The original research hypothesis tested catalog-grounded contrasts and rank-aware questions against ordinary conversational retrieval. Those optional mechanisms remain disabled in the selected release. Cycle 2 adds a bounded explicit-alternatives correctness repair after registered comparisons; it does not claim a new ranking algorithm or an incremental score gain. See [the measured result](CYCLE2_RESULTS.md).

The 30 August refinement pages from the first repeated full ranking instead of
repeating a slate that has already failed. Any ranking change returns to page
zero, and an intent override explicitly resets to page zero even when the ranked
IDs remain unchanged because earlier results may not have been eligible to
score. Intent rule coefficients are fitted on grouped independently authored language, but
intent-routed retrieval remains disabled because it added cost without changing
downstream recommendations. See [the refinement evidence](PIPELINE_REFINEMENT_RESULTS.md).

## Runtime

1. `reset` creates isolated, bounded session state. Coarse profiles are retained but not promoted into explicit shopping intent.
2. A deterministic parser records positive, negative, and neutral assertions with source turns. Known facets and conservative open-vocabulary phrases form a rebuildable query. Correction handling is attribute-local; previous recommendations are not excluded merely because another turn arrives.
3. Porter-stemmed, field-weighted SQLite FTS5 retrieves broad lexical matches and an optional category-scoped route. The broad route remains available as a rescue path.
4. Optional pinned local BGE embeddings add dense candidates through reciprocal-rank fusion. An always-on constraint guard demotes confirmed contradictions before candidate truncation and after reranking. Missing or mixed facts remain unknown and do not eliminate a product. Positive evidence boosts are separately optional. Free text is soft evidence, not a guaranteed product attribute.
5. Optional neighbor contrasts reward stated preferences that distinguish a product from a bounded lexical neighborhood. Optional MiniLM cross-encoding reranks a bounded prefix. Raw semantic logits are never claimed to be target probabilities.
6. An independent question policy selects `other`, a schedule, entropy, or bounded rank-value lookahead. Slate width remains separately configurable. Default and final-turn behavior show up to ten candidates.

The portable entry is `agent.py`; `starter/agent.py` only re-exports it for the official evaluator. Runtime code does not import the experiment runner or read target labels.

## State, selected repair and optional research modules

### Reversible source ledger

The ledger retains unrelated facts when one preference changes. Simple per-attribute latest-value and normalized-history modes provide controls. Unknown answers are not exclusions; explicit no-preference can clear a prior slot. Descriptive spans retain rare terms without copying the entire conversation into retrieval.

Corrections within a message are applied in source-clause order. Dependency invalidation happens at each correction, so reintroducing a category later in the message does not silently restore its old quantity. Product matching distinguishes direct negation, support, mixed statements and missing evidence; this is a bounded English rule system, not a formal guarantee about arbitrary prose.

Lexical no-show designs are retained as positive descriptors rather than mistaken for a product exclusion. Generic discourse such as “matters” and “use your judgment” is not a product value. An unanswered broad `other` question preserves earlier explicit details; explicit no-preference for a named color or material can still clear that specific slot. Direct catalog avoidance instructions count as negated evidence, not affirmative support.

This is a conservative English parser, not unrestricted language understanding. Complex negation, implicit corrections, multilingual requests and unrestricted Boolean expressions need additional validation. Broad profile personalization is intentionally not inferred from aggregate preferences.

### Explicit alternatives in the selected release

Known positive values connected by a direct same-attribute “or” form one choice set. The constraint guard penalizes that set only when every live option has observed contradictory evidence. Support for one option satisfies the set; any unknown option prevents a proven-contradiction claim but does not certify product fit. Independent negative requirements and ordinary conjunctions remain separate.

Explicit corrections and selections operate in clause order against live state. Rejecting one option preserves the survivor, while replacing the choice set retracts stale memberships. Group changes invalidate the ranking cache even if query words stay the same. Independent additions and exclusions survive unsupported overlapping lists. Nested, cross-attribute and body/component expressions remain outside the claim.

`alternatives_mode` provides OFF, parse-only and grouped controls. Grouped is selected under unchanged model, candidate, token and question budgets. Developer regressions and the labeled invented-catalog replay demonstrate the repair; target scores and locked capability outcomes are identical across controls. Two body/component failures remain in validation. The [protocol](CYCLE2_ALTERNATIVES_PROTOCOL.md) and [selection receipt](cycle2-selection.json) define the release boundary.

### Grounded neighbor-contrast sidecar

Uniform title-token postings produce a bounded candidate neighborhood for every catalog row. Weighted title overlap selects up to eight neighbors. A sidecar stores only facts already present in the product evidence and their support frequency among neighbors. No generated descriptions or artificial distinctions are added. Catalog-identical rows keep distinct IDs and no unsupported difference is claimed between them.

This measures lexical-neighborhood distinctiveness, not global uniqueness or user desirability. Neighbors are approximate and catalog-text evidence can be noisy. The module remains disabled unless its measured benefit justifies the extra build/storage/runtime cost.

### Rank-value question lookahead

At most forty candidates and four grounded answers per field are simulated. Rank-derived weights are explicitly heuristic. Unknown metadata, no preference, unmodeled answers and a twenty-percent outside-pool allowance retain nonzero uncertainty with no invented recovery gain.

The objective combines expected reciprocal-rank recovery and hit recovery beyond the slate already shown this turn. Asking does not consume a separate recommendation opportunity. This is a one-step heuristic, not a calibrated target posterior or proof of optimal questioning. Ordinary `other` and schedule policies are mandatory controls.

Optional gap/lookahead slate shortening is isolated because the final organizer policy may differ. It never automatically returns an empty list; explicit empty-slate experiments are bounded to one consecutive turn and recover to a full slate after an uninformative reply. The final turn restores all available capacity.

The simple `other` policy has a configurable question cap. It asks again only following informative replies and stops after an uninformative/no-preference answer. The cap and reranker settings are tuned on development sessions only.

## Reliability and resource boundaries

- Maximum 256 live sessions; least-recently-used eviction clears cached rankings too.
- Current message truncated to 8,000 characters. Search terms, dense context, model sequence length and candidate counts are bounded.
- Repeated unchanged state reuses the ranking. Cache keys include the rebuilt query and state revision.
- Optional assets are verified against catalog, model, document-view and file hashes. Model loading is local-only, safetensors-only and does not permit remote code.
- Absent assets, model errors, malformed scores and raised inference timeouts use tested sparse/evidence fallback paths. A stalled native inference kernel is not forcibly interrupted by an in-process watchdog; organizer hard timeouts remain an external constraint.
- Optional usage counts reflect actual local model input tokenization; no text-generating model is used. Sparse execution reports zero model tokens.
- No paid services, runtime network, credentials, hidden-label joins, purchase-history reconstruction or target lookup tables.

## Evaluation separation

The official evaluator remains unchanged. The experiment wrapper sees visible inputs/outputs and captures diagnostics; only after the session ends does evaluator-side analysis join target IDs to candidate ranks. Development and reserved sessions follow the frozen protocol in `EXPERIMENT_PROTOCOL.md`.

Ever-retrieved recall is policy-dependent because questions change future user disclosures. It is useful for diagnosing retrieval versus ranking loss, not a clean fixed-query comparison. Paired bootstrap uncertainty is reported alongside engineering gains. Public-session performance is not a claim about real purchase conversion or the private evaluation set.
