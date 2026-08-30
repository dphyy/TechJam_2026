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
