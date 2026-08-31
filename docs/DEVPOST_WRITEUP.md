# Mercury — submission write-up

Current pipeline and verified aggregate only. Publication, video and submission status follow the [release checklist](RELEASE_CHECKLIST.md).

## Inspiration

Shopping rarely starts with a perfect search query. Someone might begin with “I’m looking for a jacket,” discover that waterproofing matters, and later decide that a lighter material is more important. A useful shopping assistant needs to follow that conversation without forgetting earlier requirements or clinging to preferences the shopper has withdrawn.

We built Mercury around this idea: conversational search should remember what matters now, ask questions that help, and make progress when the first recommendations are not enough.

The problem statement asks for an agent that supports both targeted buying and open-ended browsing, accumulates information, and handles changing intent. Our approach combines structured conversational memory, lexical retrieval, adaptive recommendations, and guarded paging. It makes the conversation adaptive while keeping the search engine local, deterministic, and inspectable.

## What it does

Mercury is an offline conversational shopping backend that searches a catalog of 50,000 products and returns up to ten ranked recommendations per turn.

It turns the shopper’s messages into structured evidence: product category, requirements, preferences, exclusions, and corrections. Each piece of evidence records its source turn. When the shopper changes a preference, Mercury updates the active evidence while preserving unrelated requirements. “No preference” is handled as a boundary, so the assistant can stop asking about an attribute the shopper does not care about.

For example, a shopper can ask for a black leather bag with an adjustable strap, switch to blue canvas, and then remove the color preference. Mercury updates the material and color requirements while retaining the bag category and strap requirement.

Mercury also adapts how much it shows. It can offer a small shortlist when the evidence is strong or another useful answer could narrow the search, then broaden the recommendations when uncertainty remains or the conversation is running out of turns.

When the shopper’s preferences remain stable and the leading candidate set repeats, guarded paging selects highly ranked products that have not yet been shown. A preference change or explicit override resets that exploration so the best matches for the revised request are considered again.

The current default requires no hosted model, external search service, API key, or GPU. It incurs no model API charges; ordinary hardware and operating costs still apply.

## How we built it

We implemented Mercury in Python as a modular pipeline:

**Conversation → structured evidence → lexical retrieval → constraint-aware ranking → clarification and adaptive shortlist → guarded paging → validated response.**

**Conversation state.** A deterministic parser handles additions, replacements, exclusions, alternatives, and no-preference replies. It distinguishes feedback about the displayed recommendations from actual product requirements. This helps prevent phrases such as “Those options aren’t right” from becoming search terms. Source-linked diagnostics record how the active evidence changes.

**Lexical retrieval.** We use SQLite FTS5 and field-weighted BM25, alongside an exact-constraint index derived from the catalog. Retrieval combines broad keyword matches, distinctive phrases, category matches, and searches requiring multiple stated constraints. Weighted reciprocal-rank fusion combines the lexical result lists, while the exact index supplies additional candidates with matching catalog evidence.

The backend can open a validated prebuilt local SQLite index or build an in-memory index from the catalog. Product-feature caching avoids repeatedly processing the same metadata.

**Ranking.** Transparent Buying and Browsing policies adjust the emphasis placed on requirements and discovery. The current ranking first protects product-category relevance, excludes known category mismatches, and retains unknown taxonomy as a fallback. Within those category tiers, it considers explicit contradictions, exact requirement matches, phrase coherence, and field-level evidence before lower-priority score signals such as reviews and price.

Missing metadata is not automatically treated as a contradiction, although products with stronger supporting evidence can rank higher. Component-aware matching also helps distinguish requirements such as a leather upper from those concerning a shoe’s lining. The ranked context retains up to 100 candidates.

**Clarification and shortlist selection.** The question planner examines differences among the retrieved products and estimates which question could reduce uncertainty. It combines information gain with an answerability heuristic and discounts repeated questions. Early open-ended prompts allow shoppers to disclose important details that predefined attributes might miss. The shortlist policy considers ranking confidence, requirement coverage, question value, and remaining turns.

**Guarded paging.** Paging operates on the ranked context without changing retrieval scores. When preferences are unchanged and the top-ten candidate membership remains stable, it favors unseen products. It preserves the chosen shortlist size and does not increase the shortlist’s count of known constraint violations. Preference changes and explicit overrides reset exposure history. Identical request retries return the cached response without advancing the page.

**Development tools:** Python 3.13.5 for current verification (runtime requires Python 3.10+ with SQLite FTS5), a local virtual environment, Git, command-line evaluation scripts, Ruff, Python’s `unittest`, and HTML/JSON diagnostic replays.

**APIs:** The competition’s Python Agent interface, including `reset` and `respond`, and its local evaluator. The active lexical-plus-paging pipeline makes no external API calls.

**Libraries and frameworks:** SQLite FTS5 through Python’s `sqlite3`, plus standard-library modules for parsing, data structures, scoring, caching, serialization, and checksums. Earlier experiments used NumPy, SciPy, scikit-learn, PyTorch, Hugging Face Transformers, and Sentence Transformers. Those optional experimental dependencies are separated in `requirements-research.txt`, but neural reranking and vector inference are not part of the current default pipeline.

**Datasets and assets:** The competition’s frozen 50,000-product catalog, derived from the Clothing, Shoes and Jewelry category of Amazon Reviews 2023 by McAuley Lab at UCSD; 200 public development sessions; locally generated synthetic conversation sets; and authored regression cases. We use text and structured metadata, including categories, features, details, prices, and aggregate ratings. Local search indexes and diagnostic replays are derived assets. We do not use organizer-private evaluation labels or product imagery in the runtime.

## Challenges we ran into

**Following corrections without losing context.** A shopper may change one attribute while keeping everything else. We needed to distinguish a replacement from an additional preference, an exclusion, an alternative, or an explicit absence of preference.

**Separating similar products.** Catalog items often share almost all their wording. Rare phrases, exact constraints, and the fields in which those phrases appear can matter more than broad semantic similarity. We explored neural and vector approaches, but selected a default built around lexical evidence and deterministic ranking.

**Handling incomplete and misleading metadata.** A missing material is different from a stated incompatible material. Matching words can also appear in unrelated product categories. These cases motivated explicit contradiction handling, category safeguards, and conservative treatment of unreliable prices.

**Making progress without skipping the right answer.** Repeating a shortlist wastes turns, but blindly advancing after a correction can skip the newly relevant products. Guarded paging balances exploration with resets when intent changes.

**Keeping evaluation claims honest.** Public and synthetic sessions are useful development evidence, but they do not establish real-world shopping performance. We recorded source hashes, configurations, comparisons, and dataset exposure so results remain attached to the versions that produced them.

## Accomplishments that we're proud of

We built a working conversational search pipeline that runs locally, adapts to changing preferences, explains its decisions through diagnostics, and explores beyond its first recommendations without requiring model inference.

The latest source-bound public evaluation recovered **200/200 targets**, with
**MRR 0.965048**, **MTTC 2.105**, and **TechnicalScore 0.967414**. It recorded
**zero model tokens, zero agent errors and zero fallback turns**. The current
repository verification has **1,164 passing tests**, including **130 checks with
site-packages disabled**. See [the report](../REPORT.md) and
[exact source/configuration receipts](current-results.json).

These are consumed public development results, not organizer-private scores or
measured real-user outcomes. The already consumed robustness-v1 final set was not
reopened for the current cleanup. Earlier scores and repeat-reduction comparisons
are confined to the [historical experiment records](RESEARCH_INDEX.md).

## What we learned

Conversational search quality depends on more than retrieval accuracy. Remembering a correction, asking a useful question, and choosing which candidates to show can change whether the shopper ever sees the right product.

We also learned that model complexity is not a substitute for understanding the data. In this catalog, exact phrases and structured attributes carry substantial signal. A carefully designed lexical system can make strong use of that evidence with low operational complexity.

Finally, retrieval and presentation need separate evaluation. A product can already be in the candidate pool yet remain invisible because the assistant keeps showing the same shortlist. Paging addresses that problem, but its benefits must be tested alongside relevance, intent changes, and the possibility of showing a worse-ranked match.

## What's next for Mercury

The integrated pipeline has been evaluated and its documentation and demo bind to that source. Remaining submission work is to verify public repository visibility, finalize the video and contribution record, and complete the external release checklist. Any further runtime change would require a new source-bound verification.

Beyond the competition, we want to test Mercury with real shoppers, including conversations where users ignore suggestions, partially reject a list, or change direction without explicit correction phrases. Continued conversation should not automatically be interpreted as rejection in a real storefront.

We also plan to improve multilingual and paraphrase handling, with any optional language model constrained by explicit requirements and an offline fallback. For larger, changing catalogs, the next steps are incremental indexing, updated inventory and prices, and measured concurrency and memory limits.

Longer term, we want to evaluate satisfaction, relevance, and time saved alongside exact-product recovery, while making any persistent personalization opt-in and easy to inspect or delete.

## Team contributions

The five team members and their contribution roles are recorded in the [README](../README.md#team-member-contributions).
