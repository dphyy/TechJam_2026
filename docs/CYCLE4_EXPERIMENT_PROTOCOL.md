> **Historical experiment protocol — not current release guidance.** Retained to explain the controls, evaluation order and decision gates behind the improvement comparisons. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Cycle 4 score-improvement registration

Registered 29 August 2026 after the Cycle 3 candidate-admission, fielded-retrieval,
document-serialization, and bounded-depth routes all failed their promotion gates.
The objective remains a generalizable improvement to the Track 4 shopping agent, not
an attempt to reconstruct target IDs or optimize an already-consumed public result.

## Control and evidence boundary

The control is `configs/selected.json` after the independently reviewed
generalization changes already on `main`. Its source-matched Cycle 3 screening
parity result is `0.811786` TechnicalScore, `0.925000` HitRate@10, `0.635119`
MRR, and `3.062500` MTTC on the existing 160-session screening pack.

That pack is development evidence: it was generated before Cycle 3 source work,
but its outcomes have now been opened. Cycle 3 confirmation and validation remain
unopened. A Cycle 4 arm may use screening for rejection or selection only; it earns
one confirmation comparison only after it clears every screening gate below.

All arms preserve the frozen catalog, evaluator, selected local cross-encoder,
120-candidate ceiling, 256-token pair ceiling, Top-10 output limit, and no-network,
no-paid-service policy. Runtime code must not inspect target IDs, sample IDs, labels,
future turns, or simulator state.

## Failure model

The source-matched control has two sessions that never enter the 120-item candidate
pool and ten that are retrieved but fail to reach the final slate. Earlier D60 and
D120 experiments showed that expanding the reranker prefix alone is insufficient:
extra tail recall can displace existing hits. Cycle 4 therefore tests interpretation
and ordering mechanisms separately before any combination.

## Registered arms

### A. Structured state-to-reranker query

The registered candidate keeps the current 30-candidate prefix and head document. It
changes only the first element of each cross-encoder pair from a flat value join to a
deterministic request record: active category, hard requirements, soft preferences,
exclusions, and supported component qualifiers. It contains only state already derived
from customer turns. The control remains the byte-equivalent flat query.

The mechanism is distinct from Cycle 3 document serialization, which varied product
documents rather than the query. It may improve rank ordering but must not invent
facts or turn uncertain evidence into a hard constraint.

This arm was rejected on screening and its live configuration was removed; see
[the Cycle 4 screening results](CYCLE4_SCREENING_RESULTS.md). Its complete source
snapshot and configuration are retained inside the ignored evaluator run record.

### B. Stability-protected 60-candidate reranking

This arm scored the existing first 60 candidates while protecting the baseline
30-prefix slate from weak tail promotions using a fixed, query-local margin. It
improved screening TechnicalScore by only `0.000736`, well below the registered
`0.010000` gate, so its runtime implementation and configuration were removed. See
[the Cycle 4 screening results](CYCLE4_SCREENING_RESULTS.md). It must not be combined
with later arms.

### C. Exact source-phrase sparse route

This arm formed a bounded FTS route from active source clauses and fused it with the
unchanged broad and category-scoped routes. It did not recover either candidate-pool
miss in its causal preflight, because many catalog rows shared the same template
feature text. Its implementation and configuration were discarded before score
measurement; see [the Cycle 4 screening results](CYCLE4_SCREENING_RESULTS.md). It must
not be combined with later arms.

### D. Component-qualified evidence

The fixed candidate is `configs/cycle4_role_evidence.json`. It adds a bounded soft
score only when the active ledger contains both a positive matching material and an
explicit material-plus-whole-role phrase, and that same direct phrase appears in one
ordinary catalog field. A material correction or neutralization retracts support even
when the broad open-vocabulary phrase remains in state. The only accepted whole roles
are `body`, `outer shell`, `exterior`, `shell`, and `band`; component-only phrases such
as a leather handle, lining, cuff, collar, accent, insert, strap, or patch remain
unknown. The candidate never changes retrieval, model calls, output size, or hard
constraints.

Fresh authored mini-catalog tests must demonstrate a local source-span witness for the
whole product, no boost for component-only or cross-field evidence, safe behavior after
correction/neutralization, and deterministic diagnostics. Only after those tests pass
may this one fixed configuration enter screening. Missing or ambiguous evidence must
remain unknown, never a hard exclusion.

### E. Composition-qualified evidence

The fixed candidate is `configs/cycle4_composition_evidence.json`. It applies one
post-reranker soft adjustment of `+0.010` only when an active positive `other` ledger
fact is exactly `N% material`, is explicitly dependent on that same active material,
and the exact phrase appears in one ordinary catalog field. The adjustment is a single
cap even if a product has multiple supporting phrases. It never expands retrieval,
changes a model pair, adds a question, changes output length, or creates a hard
constraint.

The arm is restricted to the conversation state's existing dependent quantity facts;
it does not infer a composition from a material mention, nearby tokens, or multiple
fields. A material correction, neutralization, negative request, inactive dependent
fact, absent phrase, and cross-field co-occurrence must each receive zero support.
Direct source-span diagnostics and deterministic ordering are required. This arm may
not be combined with D in this campaign.

Its target-blind activation preflight observed 53 of the 160 released screening
sessions and 105 fresh state revisions with an active qualified fact. Every affected
current 30-item reranker prefix had at least one direct witness, and 100 prefixes had
non-uniform support; this establishes an opportunity to change order, not a score
improvement. Only after the stated safety tests pass may the fixed configuration enter
one screening comparison against the current `0.817786` source-matched control. The
one registered comparison was completed and rejected below the promotion gate; it
must not be retuned or combined with another arm.

### F. Source-alias sparse parity

The fixed candidate is `configs/cycle4_source_alias.json`. The ledger currently
canonicalizes parser-owned values before `SessionState.query()` reaches the sparse
index: a shopper's `trainers`, for example, becomes `sneakers`. This arm recovers only
the exact non-canonical parser alias that appears in the current active fact's own
source message. It never adds an external synonym, inferred relation, old source text,
negative fact, neutral answer, or dependent quantity.

When at least one such source alias exists, the unchanged broad sparse route and its
existing category-scoped route are retained, and a separate alias-only sparse route
uses the same index and `sparse_limit`. Its fixed reciprocal-rank fusion weights are
`0.850` broad plus `0.150` alias without a category, or `0.595` broad, `0.255` scoped,
and `0.150` alias with a category. With no source alias, route construction and
weights are byte-for-byte the selected broad behavior. The arm preserves the
120-candidate ceiling, the 30-pair reranker, model inputs, question policy, output
length, hard constraints, and no-network policy.

Before screening, fresh authored catalog cases must prove that a product containing
only the shopper's recognized alias can enter the alias route, while canonical-only
input, a canonical source wording, inactive, negative, neutral, and dependent facts
produce no alias route. A target-blind preflight must also establish that at least one
current screening state has a nonempty alias route and that it changes a candidate
prefix; otherwise no scorer run is justified. If both gates pass, the one fixed
configuration may receive exactly one source-matched screening comparison against the
current selected source. It cannot be tuned, combined with a rejected arm, or promoted
without clearing every registered gate below.

The preflight found active aliases on 123 of 471 recorded shopper turns and changed
the retrieval prefix on 110 turns. The one registered screening comparison was then
completed and rejected: it left TechnicalScore, HitRate@10, MRR, MTTC, and efficiency
unchanged. The configuration is not selected and must not be tuned or combined.

## Screening and promotion rule

For every arm, run a source-matched selected control and exactly one configured
candidate on the Cycle 3 screening pack. An arm earns confirmation only when all of
the following are true:

- TechnicalScore improves by at least `0.010000`;
- HitRate@10 declines by no more than `0.010000`;
- no scenario score declines by more than `0.020000`;
- there are no invalid IDs, source drift, fallbacks, or agent-error turns;
- warm p95 is no more than twice its matched C0 measurement; and
- the stated causal diagnostic improves: rank distribution for A/B, candidate recall
  for C, or paired source-qualified cases for D/E.

Only one qualified survivor may consume the Cycle 3 confirmation pack. The source,
configuration, target-lock verification, and model hashes must be frozen first. A
nonnegative confirmation delta with no correctness or resource regression is required
before one final validation comparison. No public 200-session run selects, repairs,
or relabels a Cycle 4 contender.
