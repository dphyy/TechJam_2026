# Intent-conditioned ranking experiment

Registered 30 August 2026 before score-bearing runs. This branch tests one fixed
policy against the locked finalist on the released 200-session public set.
Because that set has been repeatedly examined, its result is descriptive
development evidence and cannot establish likely private-set improvement.

## Fixed policy

- `buying`: apply a `0.10` bounded score adjustment using positive/negative
  catalog evidence for typed hard, non-price requirements, then reapply the
  existing hard-contradiction guard. Unknown evidence remains neutral.
- `browsing`: apply the existing deterministic facet-diversity reranker at
  strength `0.20` inside the first 30 non-penalized candidates. The leader stays
  anchored and no candidate crosses a hard/object guard boundary.
- `mixed`: preserve the finalist ranking unchanged.

The weights are inherited from the already recorded typed-plan active trial and
intent-diversity oracle; there is no public-set sweep. Routing uses only the
existing live `IntentDecision.mode`, with no evaluator scenario label, target,
sample ID, or future message available to runtime.

## Measurement

Run a source-matched disabled control and the one enabled candidate on identical
public bytes and local model assets. Report official aggregate and scenario
metrics, paired session changes, action activation by evaluator scenario,
classifier routing, changed positions, hard-evidence adjustments, facet
coverage, latency, tokens, and fallbacks. A small public gain is a reason for a
new independent downstream pack, not direct finalist promotion; a material
regression rejects the policy.

## Public-set result — reject the composite policy

| Variant | HitRate@10 | MRR | MTTC | TechnicalScore | Tokens | warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| Disabled control | 0.985000 | 0.646718 | 2.860000 | 0.849315 | 2,375,008 | 1.123507 s |
| Intent-conditioned | 0.985000 | 0.642141 | 2.855000 | 0.848042 | 2,375,008 | 1.196009 s |

The paired TechnicalScore delta is `-0.001273` with a 10,000-resample 95%
bootstrap interval of `[-0.003201, +0.000015]`. HitRate is unchanged, MRR falls
`0.004577`, MTTC improves only `0.005`, and both arms have zero fallbacks. Ten
session outcomes change: three reciprocal ranks improve, seven worsen, no hit
is gained or lost, and one intent-override target arrives one turn earlier but
at a substantially worse rank.

Scenario TechnicalScore deltas are boundary `0.000000`, browsing `+0.000089`,
buying `-0.000814`, and intent override `-0.006556`. The intended separation is
not achieved by the live router:

- Across true buying turns, 137 are classified buying but contain no parsed hard
  constraint, zero receive hard-evidence ranking, 18 are routed to browsing
  diversity, and 34 remain mixed.
- Across true browsing turns, 74 receive diversity, 97 are classified buying
  without a hard constraint, one receives hard evidence, and 29 remain mixed.
- Of 56 total hard-evidence activations, 54 occur in intent-override turns, one
  in browsing, one in boundary, and none in true buying turns.
- Diversity changes 75 true-browsing turns, but browsing score remains
  effectively neutral; it also changes 18 true-buying turns.

This public set is already heavily consumed, so even a gain would not establish
private-set transfer. The observed slight regression plus severe action-routing
mismatch provides no reason to adopt this implementation. Reject the composite
policy and keep `intent_conditioned_ranking` disabled. The experiment is useful
because it identifies the prerequisite for a credible retry: first improve and
independently validate intent/action routing and hard-requirement extraction,
then test buying evidence and browsing diversity as separate arms on a new
target/user-disjoint pack rather than coupling them immediately.
