# Scoring and judging

The unchanged [official evaluator](../evaluator/local_evaluator.py) searches for
one exact target ID per session. It receives actual responses from `agent.Agent`.
The current public pipeline is lexical retrieval with adaptive shortlist and
guarded paging; see [README](../README.md).

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 − MTTC) / 10, 0, 1)
```

- HitRate@10: fraction of sessions with the target in an eligible scored slate.
- MRR: reciprocal target rank on the first eligible hit; misses contribute zero.
- MTTC: average first eligible hit turn; a miss counts as turn 11.
- Intent-override eligibility follows the unchanged simulator/evaluator gate;
  finding a target before that gate is not necessarily a scored success.

A score of 0.96 does not mean 96% accuracy, 96% of the human rubric, or a percentile
among teams. Recommendation score magnitudes and prose are not direct formula
terms. Questions affect what the simulator reveals. Shorter slates can improve
first-hit rank while sacrificing recall; paging can recover later targets but
can delay a previously shown one. Legality, latency, repeat behavior and
constraint compliance therefore need separate checks.

[Current-results.json](current-results.json) and [the report](../REPORT.md) name
the current measured source and evidence boundary. Earlier lexical paging and
neural scores are historical, not interchangeable with current measurements.
Public development data and previously opened synthetic datasets are consumed;
the robustness-v1 final pair has already been evaluated. This repair does not
use that final set for tuning or rerun it. See [dataset status](DATASET_STATUS.md).

Human judging is separate. The supplied Track 4 brief uses 35/20/20/15/10 planning
weights, while the general Stage 2 rules specify four equal criteria. Preserve
that distinction rather than selecting whichever yields a higher self-grade.
See [rubric context](CHALLENGE_RUBRIC_CONTEXT.md) and [official event rules,
checked 31 August 2026](https://tiktoktechjam2026.devpost.com/rules).

The event deadline displayed on the rules page is 1 September 2026, noon SGT.
The public repository/README, project description and additional track deliverables
are separate release requirements. A strong local score is not a completed or
eligible submission; see [release checklist](RELEASE_CHECKLIST.md).
