# Challenge rubric and current runtime

The supplied Track 4 brief concerns exact hidden-product recovery over the
50,000-product catalog through at most ten conversational turns. Buying,
browsing, overrides, and no-preference responses are distinct evaluation cases.
No target IDs, scenario labels, or evaluator outcomes are runtime features.

The public implementation is **lexical search with adaptive shortlist and guarded
paging**, defined by `mercury.lexical.config.DEFAULT_AGENT_CONFIG`. D30/MiniLM,
review-prior, dense-admission and fusion results describe research configurations,
not the public default. See [design](DESIGN.md) and [research index](RESEARCH_INDEX.md).

| Supplied brief criterion | Brief weight | Current evidence and limits |
|---|---:|---|
| Technical execution | 35% | Legal API, transactional state/paging, source-bound evaluation, regression tests; parser and metadata limits remain |
| Innovation and problem insight | 20% | Inspectable evolving intent and guarded exploration; no claim to invent lexical retrieval or paging |
| Impact and relevance | 20% | Exact-target recovery and repeat reduction are proxies, not demonstrated user satisfaction or conversion |
| Feasibility and practicality | 15% | Offline standard-library runtime; organizer hardware/resource limits still need verification |
| Presentation and communication | 10% | Current README and actual public-agent demo; video/publication/submission remain separate gates |

These planning weights are from the supplied brief, not a guarantee of how the
human judges aggregate scores. The general event's Stage 2 rules specify four
equally weighted criteria: technical execution, innovation/problem insight,
feasibility/practicality, and impact/relevance. Consult the applicable judging
stage and organizers if the brief and event material differ. [Official rules,
checked 31 August 2026](https://tiktoktechjam2026.devpost.com/rules).

The evaluator's `TechnicalScore` is a separate formula, not the 35-point human
rubric. See [scoring guide](SCORING_AND_JUDGING.md). Current evidence must come from
the current source receipt; historical high scores cannot grade changed code.
Use [dataset status](DATASET_STATUS.md), not a filename containing “sealed,” to
assess whether evidence was held out.
