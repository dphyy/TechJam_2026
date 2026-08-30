# Bounded continuation and question-policy results

Phase 12 evaluated two independent, opt-in mechanisms against the frozen Phase
10 D30 control. Neither arm was combined with adaptive depth or the rejected
domain reranker.

## 12A: Explicit-rejection continuity

The candidate keys continuity to the canonical active-state signature and the
exact previously returned slate. Under the same intent, an explicit item-slate
rejection excludes only those exact ten IDs from the current ranking, tolerating
harmless full-order jitter. Filler and no-preference answers return to page 1;
an active fact change or override resets immediately. Tests cover explicit
rejection, filler, fact changes, and override behavior.

The screening run is
`runs/phase12a-rejection-continuity-screening-20260830`. It lost one buying hit:
overall HitRate fell from `0.98125` to `0.975`, buying HitRate fell from
`0.986667` to `0.973333`, and TechnicalScore fell from `0.865071` to
`0.862694`. It is rejected before confirmation.

## 12B: Conservative discriminating question

The candidate asks a typed attribute only when at least 60% of the Top 30 has
catalog facet evidence, at least two values are represented, the second group
covers at least 15% of rows, and the largest supported group is at most 75%.
It asks at most one typed facet per session, never treats missing metadata as a
value, and otherwise uses the existing bounded `other` fallback.

The screening run is
`runs/phase12b-discriminating-question-screening-20260830`. It produced 120
well-supported typed questions and reduced repeated question attributes from
118 to 67, but productive previous answers fell from 87 to 79 while neutral
answers rose from 48 to 70. HitRate stayed `0.98125`, but MRR fell from
`0.655238` to `0.616019`, MTTC worsened from `2.10625` to `2.15625`, and
TechnicalScore fell to `0.852306`. Warm p95 also increased materially. It is
rejected before confirmation.

Both configurations remain reproducible negative-result artifacts. The Phase 10
D30 admission candidate with the existing paging and `other` question policy is
the strongest continuation candidate.
