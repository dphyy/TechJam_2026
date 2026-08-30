# Track 4: score, limits and judging

Verified 26 August 2026 against participant release `9a35be51780ff1caf89eceaabca34259e946f40f`, the recorded official results and the event pages. The upstream `main` branch still points to that release. This is analysis of a frozen preparation build, not a new benchmark or a competition submission.

## The task in ordinary language

Build a shopping assistant's search backend. A simulated shopper has one hidden target among 50,000 clothing, shoe and jewelry products. The assistant receives messages and an anonymized aggregate profile, asks follow-up questions, remembers changing preferences and recommends up to ten catalog IDs per turn. It has at most ten turns. Success means finding the exact hidden ID early and near the top, not merely recommending a plausible substitute.

The four scenarios are buying, browsing, intent changes and no-preference boundaries. There are 200 public development sessions and 800 organizer-held private sessions. This is simulated target recovery, not measured real-world purchasing or revenue. See the unchanged [competition specification](competition_specification.md) and [evaluator](../evaluator/local_evaluator.py).

The technical deliverable is a Python `Agent`, helper modules, setup instructions and a method/resource report. A frontend is not required. The event additionally asks for written submission material, a public repository and a public three-minute YouTube demo; an API/results walkthrough fits the supplied backend-track brief. [Event requirements](https://tiktoktechjam2026.devpost.com/), [backend submission contract](submission_rules.md).

## Current selected result: 0.844994

The 31 August realistic-shopping merge reproduced the following result on all 200 released public development sessions:

| Component | Current result | Weight |
|---|---:|---:|
| HitRate@10 | 196/200 = 0.980000 | 50% |
| MRR | 0.640647 | 30% |
| Efficiency | 0.814000, from MTTC 2.860 | 20% |

```text
TechnicalScore = 0.50 × 0.980000 + 0.30 × 0.640647 + 0.20 × 0.814000
               = 0.844994, rounded to six decimals
```

This is consumed public development evidence. It is neither a private-test forecast nor 83.92% accuracy; HitRate@10 is the accuracy-like 97% value. The complete aggregate and claim boundary are in [current-results.json](current-results.json).

## Historical fixed-slate result: what 0.786724 meant

| Component | Definition | Selected whole-public result | Weight |
|---|---|---:|---:|
| HitRate@10 | Fraction of sessions with an exact target in the scored Top 10 within ten turns | 179/200 = 0.895000 | 50% |
| MRR | Mean reciprocal target rank on the first successful slate; misses contribute zero | 0.613746 | 30% |
| Efficiency | `clip((11 - MTTC) / 10, 0, 1)`; misses count as turn 11 | 0.775500, from MTTC 3.245 | 20% |

```text
TechnicalScore = 0.50 × 0.895000 + 0.30 × 0.613746 + 0.20 × 0.775500
               = 0.786724, rounded to six decimals
```

It is not 78.67% accuracy, a judge rating or a percentile among teams. HitRate is the 89.5% figure. Latency, model tokens and cost are disclosed feasibility measures, not terms in this formula. Only exact valid unique IDs count; prose quality and an optional recommendation score do not directly change the numeric score. Follow-up choice does matter because the simulator uses `ask_attribute` to decide what information to reveal. Sources: [scoring configuration](evaluation_config.json), [official evaluator](../evaluator/local_evaluator.py).

The pinned evaluator gives a standardized comparison when agents use the same catalog, sessions, simulator and metric version. Different models or random agent behavior can still produce different runs. Final hardware restrictions, evaluator revisions and the score's mapping into human judging require organizer confirmation. Our public score is not a private leaderboard result.

## Why 1.000000 is not the right target

On this exact public set, even a target-knowing oracle cannot reach 1.000000 without breaking the unchanged evaluator's intent-override gate. Among 200 sessions, 170 permit a hit on turn 1, twelve first permit it on turn 3 and eighteen on turn 4. If an oracle always ranked the target first at its earliest eligible turn:

```text
Minimum MTTC = (170 × 1 + 12 × 3 + 18 × 4) / 200 = 1.39
Maximum Efficiency = (11 - 1.39) / 10 = 0.961
Gate-only score upper bound = 0.50 + 0.30 + 0.20 × 0.961 = 0.992200
```

This is a mathematical upper bound, not an achieved result or a promise that a legitimate agent can approach it. Initial requests often do not distinguish one exact item; additional turns reveal necessary information. Missing or contradictory metadata can prevent reliable disambiguation. Catalog-identical IDs also exist, though that observation does not prove they caused our current misses.

The calculation reads public behavior only for offline analysis. No oracle, gate lookup or sample-specific behavior is added to the runtime. Reproduce the bound from the original public release:

```python
from collections import Counter
from evaluator.local_evaluator import catalog_index, load_jsonl, materialize_hidden_fields

_, _, products = catalog_index("data/catalog.jsonl")
earliest = []
for sample in load_jsonl("data/public_set.jsonl"):
    _, behavior = materialize_hidden_fields(sample, products)
    earliest.append(
        int(behavior["override"]["turn"])
        if sample["scenario_type"] == "intent_override" else 1
    )
minimum_mttc = sum(earliest) / len(earliest)
print(Counter(earliest))
print(0.8 + 0.2 * ((11 - minimum_mttc) / 10))
```

Of our 196 successful sessions, 100 first hit at rank 1 and 96 at ranks 2–10; four sessions missed. Relative to the formal 1.0 reference, the weighted gaps are 0.010000 in hits, 0.107806 in reciprocal rank and 0.037200 in turn efficiency. These gaps overlap in their causes: recovering a miss improves several components at once. They are not independent predicted gains.

The present bottleneck is mainly getting retrieved candidates into the right order and obtaining useful dialogue evidence. Every target appeared somewhere in the 120-candidate pool at some point, but this is policy-dependent session recall, not proof that retrieval is solved. More model calls alone did not solve the problem: reranking 60 instead of 30 candidates produced the same reserved hits with approximately twice the neural work.

## Is the result impressive?

Yes as a measured engineering result; not yet as proof of a novel winning algorithm. The original weak starter found 25/200 targets at 0.106710. The current realistic-shopping release finds 196/200 at 0.844994; the historical fixed-slate neural release found 179/200 at 0.786724. The stronger historical development comparison is 0.699945 for corrected stateful sparse search versus 0.775118 with the selected reranker: +0.075173, paired bootstrap 95% interval [0.039850, 0.111205]. See [the full report](../REPORT.md).

The credible work is reversible preference state, conservative evidence handling, offline reproducibility, failure tests, frozen selection and openly reported failed experiments. BM25 plus a cross-encoder is established technology. The contrast and adaptive-question experiments did not justify inclusion. The 40-session reserve is public development, not an untouched private benchmark. No competitive placement or real-user conversion improvement has been measured.

## Score is evidence, not the entire competition

The supplied early Track 4 brief assigns 35% technical execution, 20% innovation/problem insight, 20% impact/relevance, 15% feasibility/practicality and 10% final-event presentation. The general rules instead describe four equally weighted Stage Two criteria and say the rules prevail over conflicting materials. The precise TechnicalScore-to-judge-score mapping is unpublished; ask the organizers which weights apply at each stage. [Official rules, sections 6 and 11](https://tiktoktechjam2026.devpost.com/rules).

Do not assume a small public-score lead guarantees a prize. Judges can assess submission media without executing the code. The published S$15,000 first prize is event-wide, not a guaranteed separate Track 4 prize. [Official rules, sections 4 and 8](https://tiktoktechjam2026.devpost.com/rules).

## The strongest next proof

Recommended direction, not work claimed complete: make reversible intent visible and prove it generalizes. The demonstration claim should be: the assistant can change its mind without forgetting what still matters, while refusing to invent missing product facts.

1. Show a real correction from black leather to blue canvas, keeping unrelated bag/strap requirements. Display old assertions being withdrawn, retained assertions, and actual returned IDs.
2. Compare the same conversation against a strong stateful search baseline. Show a real failure and recovery, not a fabricated competitor mistake or a preselected target lookup.
3. Show an unknown-price or conflicting-material case. Make supported, contradicted and unknown evidence visibly different. Product metadata must justify the explanation.
4. Finish with the unchanged-harness result, honest ablations, CPU latency and a real offline/fallback run. State public/private uncertainty explicitly.

The judge showcase now makes the backend understandable without becoming a separate storefront project. It records real outputs and exposes retained/retracted state, intent rationale, paging, fallbacks and supported/contradicted/unknown catalog evidence. Weak soft-preference matches can still appear and remain visibly unknown rather than being disguised as verified matches.

For score improvement, prioritize ranking quality, explicit negative-feedback handling and robust correction parsing on independently authored conversations. Measure against the strong baseline, with held-out evidence and latency/cost caps. The existing 40-session reserve is now consumed; it must not be presented as fresh validation for future tuning. More public-set optimization alone will weaken the credibility of generalization claims.

Preparation is dated 26 August. The official window is 29 August noon–1 September noon SGT; pre-existing work needs significant updates during that window. Preserve real timestamps and clarify eligibility before submitting. [Official rules, sections 1 and 4](https://tiktoktechjam2026.devpost.com/rules).
