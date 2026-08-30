# Development-only tuning ledger

The 160-session development split is the only tuning set. Reserved evaluation is limited by the previously registered protocol. All comparisons are exploratory until the source/config freeze and reserved run; no private-test performance is known.

## Iterations 1 and 2

The first implementation improved the sparse starter through multi-turn state and field-aware retrieval. A second parser pass preserved quantities and compound materials and stopped generic uninformative replies from contaminating queries. On iteration 2, disabling optional positive-evidence boosts improved TechnicalScore from 0.610760 to 0.677612. The compact reranker raised it to 0.706863; adding dense retrieval reduced it to 0.685234. These are separate measured configurations, not an assumption that each model helps.

Same sparse backbone, evidence boosts off: latest-per-attribute state 0.547427; history state 0.688393; answer-aware schedule 0.642476; entropy 0.665713; rank-value 0.673681; neighbor contrast 0.679245; gap/lookahead slate policies both 0.677612; no questions 0.165118. History's slight public-score gain does not justify retaining explicitly withdrawn requirements in the released product. Contrast's +0.001633 gain is below the predeclared practical threshold. Conservative slate policies did not change outcomes in this test.

## Final bounded search, declared before these runs

1. Integrate general correctness fixes found by independent review: negated product evidence, same-message correction order, input length and official-hit diagnostics. Keep an always-on, unknown-safe contradiction guard even when optional positive-evidence boosts are disabled.
2. Retest the sparse and neural reference configurations on the corrected source.
3. Test two additional `other`-question caps (4 and 9, versus the existing 2). A useful reply may justify asking again; an uninformative/no-preference reply still stops repetition. This is an ordinary answer-aware dialogue policy, not a public-target or sentence-template lookup.
4. Test reranking weight 0.50 and 0.75 against 0.25 with the same 30-candidate cap. If a higher weight helps, test the best weight once with a 60-candidate cap; otherwise stop that branch.
5. Combine only justified improvements and test contrast/rank-value on the strongest simple neural backbone before final selection. Do not explore a Cartesian product or tune to reserved outcomes.

Selection still uses practical gain, paired uncertainty, hit-rate floor, resource cost and correctness. Zero paid-service budget. Record all negative results. The final held-out public subset has only 40 sessions and cannot certify hidden-set or real-world purchase uplift.

## Iteration 3 results and final correction pass

All six runs completed without source drift, fallbacks or agent errors. Scores: sparse reference 0.660781; four-question cap 0.673656; nine-question cap 0.673656; neural weight 0.25 at 30 candidates 0.688899; weight 0.50 0.719731; weight 0.75 0.733955. The four-question cap improves sparse score by 0.012875, paired 95% CI [0.00050, 0.03113]. Nine questions adds nothing, so retain four. Weight 0.50 improves over 0.25 by 0.030831, CI [0.00704, 0.05729]; the 0.75 result justifies the predeclared one-time 60-candidate experiment.

Before freeze, an evaluator-side development failure audit found general correctness errors: lexical “no-show socks” incorrectly created a socks exclusion; direct “avoid soaking” catalog instructions were treated as support for soaking; generic metadiscourse could become a color value; and a vague no-preference reply to `other` erased previously explicit open-vocabulary details. Fix these with invented-message/catalog regressions; do not retain false exclusions or unintended retractions to preserve an earlier score. No reserved outcomes have been opened.

Confirm on the corrected source: sparse with question cap four, neural weights 0.50 and 0.75 with cap four, and weight 0.75 with a 60-candidate prefix. These are the declared combination/depth tests, not a new Cartesian search. Test contrast and rank-value once each on the strongest justified simple neural configuration, then freeze. Do not add further weight/depth tuning from reserved results.

## Iteration 4 confirmation

All four completed runs use the same corrected source, with zero fallbacks or agent errors. Sparse score 0.699945; neural weight 0.50 / depth 30 score 0.756883; weight 0.75 / depth 30 score 0.775118; weight 0.75 / depth 60 score 0.788121. Hit rates respectively 0.81250, 0.87500, 0.88750 and 0.90625.

Weight 0.75 versus 0.50 has equal runtime complexity and a +0.018235 point-score gain, but its paired 95% interval [-0.000060, 0.037485] is inconclusive. The 60-prefix comparison gains +0.013003, interval [-0.002565, 0.031420], while p95 rises from 0.351s to 0.730s and local input tokens from 1.895M to 3.942M. Both remain well within the predeclared engineering caps; this does not establish final-host compliance. Do not present either parameter comparison as certain superiority.

Use the highest-scoring ordinary system, weight 0.75/depth 60/simple questions, as the fair backbone for the final contrast and rank-value comparisons. If neither research extension justifies inclusion, freeze the 30- and 60-prefix ordinary systems for the limited final comparison. Prefer the lower-cost variant if reserved evidence is inconclusive; do not introduce another weight, depth or policy after opening reserved results.

## Finalists selected before reserved evaluation

Final contrast score 0.793321 versus 0.788121 control: +0.005199, CI [-0.004972, 0.017714], below the practical threshold despite a slightly higher point hit rate. Disable it. Rank-value score 0.761205, below the 0.788121 simple-question control with identical hit rate (0.90625) but slower conversion (MTTC 4.100 versus 3.14375). Disable it. Neither result justifies another tuning round.

Freeze `final_neural75.json` (30 candidates) and `final_neural75_60.json` (60 candidates), including full source hashes, before opening reserved outcomes. The provisional selected entry is the lower-cost 30-candidate configuration. Both retain reversible state, mandatory unknown-safe contradiction handling, field-weighted sparse retrieval, local reranking and simple questions capped at four. Dense retrieval, optional positive boosts, contrasts and adaptive question/slate policies are off. Reserved evaluation may choose between these frozen variants only; it cannot change their settings or source.

## Reserved outcome and locked release

Freeze timestamp: 2026-08-26 08:54:43 UTC (16:54:43 SGT). Each finalist was evaluated exactly once on the 40-session reserve, with matching source hashes and no fallback. Both hit 37/40 targets (0.925). The 30-prefix score is 0.833146; the 60-prefix score is 0.832833. Reserved p95 latency is 0.388s versus 0.711s; local input tokens are 481,197 versus 998,514. Select the lower-cost 30-prefix system, which was already the provisional selected entry. No settings or runtime source changed after reserve.

The whole-public offline run after this choice is descriptive reproduction, not another untouched test. Further improvement needs a newly declared evaluation protocol and fresh evidence; do not keep tuning this reserve and calling it held out.
