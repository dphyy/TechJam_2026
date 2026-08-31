# Mercury Conversational Search

Mercury is an offline Python backend for the TechJam conversational shopping search task. It receives a changing shopping conversation, asks one allowed follow-up question, and returns up to ten real catalog `parent_asin` IDs.

The repository keeps the original weak baseline for comparison and exposes the improved Mercury agent through the official `Agent` API.

## What Is Here

- `agent.py`: portable submission entrypoint. It loads `configs/selected.json` and instantiates Mercury.
- `starter/agent.py`: compatibility export used by the official local evaluator.
- `baselines/official.py`: preserved original baseline, a stateless BM25 searcher over the latest user message.
- `mercury/`: runtime implementation for state, catalog parsing, retrieval, ranking, neural reranking, and question policy.
- `evaluator/local_evaluator.py`: unchanged public-set evaluator.
- `experiments/evaluate_suite.py`: concise comparison harness for baseline, sparse Mercury, selected Mercury, and candidate configs.
- `experiments/private_like_validate.py`: authored engineering validation runner for private-like robustness cases.
- `demo/showcase.py`: generates a portable judge-facing evidence viewer from real agent turns.
- `data/private_like_capabilities.json`: small mini-catalog validation pack for known risk areas.
- `docs/`: design notes, setup details, scoring explanation, historical experiment records, and release caveats.
- `plan.md`: phased architecture and promotion gates for the intent-routed pipeline.

Future model or product work should first read the durable
[challenge/rubric alignment](docs/CHALLENGE_RUBRIC_CONTEXT.md). It maps the supplied
brief to the current implementation, selected configuration, rejected experiments,
submission obligations, and the active refinement plan without duplicating their
full histories here.

Generated data, model assets, raw runs, and local virtual environments are ignored by Git.

## Pipeline

The selected Mercury agent runs this turn pipeline:

```text
user message
-> parse active preferences
-> retract stale preferences
-> classify buying / browsing / mixed intent
-> build a typed retrieval plan
-> build query from live state
-> sparse catalog retrieval
-> optional category-scoped retrieval
-> route fusion
-> contradiction guard
-> bounded review-count / confidence-adjusted-rating prior
-> truncate to 120 candidates
-> local MiniLM rerank of the top 30 candidates
-> small review prior on the new neural score scale
-> contradiction guard again
-> apply a small soft-price preference (unknown prices remain neutral)
-> choose ask_attribute
-> return Top-10 recommendations
```

The main engineering difference from the baseline is that Mercury treats the task as a conversation. It keeps a source-linked preference ledger, handles corrections and no-preference answers, distinguishes support from contradiction and unknown metadata, and degrades to a sparse fallback if optional model assets are missing. Named attribute questions are single-use. Bounded open-vocabulary discovery can still use the `other` API facet more than once, but each visible prompt is different and an unproductive reply stops further generic questioning.

Price is deliberately soft. Exact catalog prices can provide a small bounded ranking preference, but missing, malformed, lower-bound, or otherwise inconclusive prices remain neutral and never exclude a product.

The codebase also contains independently gated pipeline-evolution experiments:

- `configs/routed_retrieval.json`: buying/browsing/mixed route orchestration.
- `configs/product_guard.json`: object/accessory and component-scope guarding.
- `configs/structured_rerank.json`: labeled cross-encoder context.
- `configs/intent_clarification.json`: intent-aware questions and broad-query cutoff.
- `configs/runtime_adaptation.json`: weak profile prior and inferred-soft-signal decay.
- `configs/sufficiency_probe.json` and `configs/sufficiency_clarify_first.json`: bounded pre-expensive-retrieval decisions.
- `configs/uncertainty_cascade.json`: one-pass D30-or-D60 compute escalation with a hard session budget.
- `configs/multi_hypothesis.json`: at most two intent hypotheses sharing one retrieval budget.
- `configs/semantic_dialogue.json`: semantic question-goal deduplication and value-gated questions.

None passed the repository's promotion rule, so they are not enabled by the public entrypoint. See [the measured evolution report](docs/PIPELINE_EVOLUTION_RESULTS.md).
The newer robustness candidates were also rejected for promotion on frozen unseen-target development evidence; see [the roadmap results](docs/ROADMAP_IMPLEMENTATION_RESULTS.md).
Margin-aware neural fusion was similarly rejected after a preregistered Cycle 5 screening loss and remains disabled; see [its result](docs/MARGIN_FUSION_RESULTS.md).
Paging from the first repeated ranking was also rejected: it improved some ordinary-session turns but hid pre-override targets that were not yet eligible to score, reducing TechnicalScore from `0.839176` to `0.829540`. See [the early-paging result](docs/EARLY_PAGING_RESULTS.md).
A guarded follow-up fixes that failure by resetting to page 1 on detected intent overrides. It preserves 0.97 HitRate and raises TechnicalScore slightly to `0.839390`, so it is selected under the registered non-decline rule. See [the guarded result](docs/EARLY_PAGING_OVERRIDE_RESET_RESULTS.md).
The merged realistic-shopping policy then adds stable Top-10 repeat detection and highest-ranked-unseen pages while clearing exposure history on semantic intent changes. It finds 196/200 public targets and raises the consumed-development TechnicalScore to `0.844994`; see [the matched result and behaviour audit](docs/REALISTIC_SHOPPING_MERGE_RESULTS.md).

The current release adds a capped mixed review-count/star-rating prior, with separate
admission and final-ordering roles. It finds 198/200 public targets at `0.866792`,
improves all three Cycle 5 splits, and slightly improves the lower-popularity Cycle 3
counter-test. Both local constraint checks remain enabled. See the
[comparison, token costs, and remaining parser limitations](docs/REVIEW_PRIOR_RESULTS.md).
A subsequent 45-setting ratio/stage-weight search used new family-disjoint
development and reserved packs, without rerunning public or old confirmations.
Count-only improved the reserved score but failed the predeclared development-gain
threshold, so the 50:50 production blend remains unchanged; see the
[fresh tuning results and data-scope limits](docs/REVIEW_PRIOR_TUNING_RESULTS.md).
Direct neural-weight tuning over `0.60`–`0.90` also found no candidate that cleared the registered practical-gain gate, so the selected `0.75` weight remains unchanged; see [the tuning result](docs/NEURAL_WEIGHT_TUNING_RESULTS.md).

Release and experiment interpretation is fixed as follows:

- Submission/reliability: the selected 30-candidate architecture with paging from the first repeated ranking and an intent-override reset to page 1.
- Public-score demonstration only: D120's recorded `0.807170`; it is not the selected release.
- Future candidates must pass a registered evaluation on new unseen sessions before promotion.

See [the post-merge decisions](docs/MERGE_DECISIONS.md).

## Setup

Use Python 3.12 if possible. The recorded environment was CPython 3.12.12 on macOS arm64.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

If `python3.12` is unavailable locally, the current checkout has also been smoke-tested with the local `.venv` Python 3.13.5, but the documented reproducibility target remains Python 3.12.

## Data

Place the organizer catalog at `data/catalog.jsonl`. The public session file is expected at `data/public_set.jsonl`.

Expected public inputs:

```text
data/catalog.jsonl      50,000 rows
data/public_set.jsonl      200 rows
```

Verify checksums:

```bash
shasum -a 256 data/catalog.jsonl data/public_set.jsonl
```

Expected hashes:

```text
da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67  data/catalog.jsonl
857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579  data/public_set.jsonl
```

Do not commit the catalog, downloaded model weights, raw traces, private data, or generated run artifacts.

## Model Assets

The selected config uses a local cross-encoder reranker:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

Prepare and verify it with:

```bash
python -m experiments.prepare_models --model reranker --download
python -m experiments.prepare_models --model reranker
```

The files should appear under `artifacts/models/reranker/`. Runtime model loading is local-only and disables remote code. If the reranker is absent or invalid, Mercury still returns legal recommendations through sparse fallback, but it will not reproduce the selected neural score.

## Quick API Smoke Test

```bash
python - <<'PY'
from agent import Agent

agent = Agent("data/catalog.jsonl")
try:
    agent.reset("demo", {
        "purchase_frequency": "occasionally",
        "average_prior_rating": None,
        "rating_style": "balanced",
        "preference_tags": [],
        "summary": "",
    })
    print(agent.respond("demo", "I need a blue canvas bag with a zipper.", 1, 10))
    print(agent.respond("demo", "Actually, no leather ones.", 2, 10))
finally:
    agent.close()
PY
```

## Evaluate

Run the official-style evaluator against the public set:

```bash
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

Run the comparison suite:

```bash
python -m experiments.evaluate_suite \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output runs/evaluation-suite
```

Add a candidate config:

```bash
python -m experiments.evaluate_suite \
  --output runs/evaluation-suite-candidate \
  --candidate my_candidate=configs/my_candidate.json
```

Recent local public-set metrics:

| Run | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Original baseline | 0.125000 | 0.068034 | 9.810000 | 0.106710 |
| Mercury sparse fallback | 0.850000 | 0.535673 | 3.745000 | 0.730802 |
| Mercury selected neural, historical fixed slate | 0.895000 | 0.613746 | 3.245000 | 0.786724 |
| Mercury neural, historical realistic-shopping release | 0.980000 | 0.640647 | 2.860000 | 0.844994 |
| Mercury selected neural, bounded mixed review prior | 0.990000 | 0.683974 | 2.670000 | 0.866792 |

The current result was measured on 31 August with 0 fallbacks and 0 agent-error turns. Public-set results are consumed development evidence, not private-test performance. The tracked [machine-readable result](docs/current-results.json) is the single current headline; historical rows remain tied to their original source/configuration.

## Judge Showcase

Generate a portable evidence viewer from real selected-agent outputs:

```bash
python -m demo.showcase \
  --results docs/current-results.json \
  --output artifacts/judge-showcase
```

Open `artifacts/judge-showcase/index.html`. The five-turn story shows a shopper changing black leather to blue canvas while retaining an adjustable strap, declining extra questions, then rejecting the current slate at the real paging boundary. Judges can inspect the structured override decision (changed attributes plus retired, added, and retained facts), routing rationale, result paging, fallbacks, real catalog IDs, and supported/contradicted/unknown evidence. The adjacent `evidence.json` contains the complete machine-readable trace. Use a new output directory for each run; the generator refuses to overwrite prior evidence.

## Private-Like Engineering Validation

Run the authored robustness pack:

```bash
python -m experiments.private_like_validate \
  --config configs/selected.json \
  --output runs/private-like-selected
```

The pack covers vague queries, intent overrides, explicit negation, no-preference replies, alternatives, body/component scope, accessory-vs-object confusion, sparse metadata, and negative feedback.

Recent selected-agent result:

```text
19 passed
0 failed
0 unverified
fallback_turns: 0
```

This is regression and demo confidence, not organizer-private evidence.

## Tests And Lint

```bash
python -m unittest discover -s tests -q
python -m ruff check .
python -m pip check
```

Recent verification:

```text
547 tests passed
ruff passed
pip check passed
```

Some tests intentionally inject optional-component failures, so warning logs about dense, neural, contrast, and constraint fallbacks are expected during the suite.

## Interpreting The Score

The evaluator computes:

```text
TechnicalScore = 0.50 * HitRate@10
               + 0.30 * MRR
               + 0.20 * clip((11 - MTTC) / 10, 0, 1)
```

HitRate measures whether the exact hidden target appears in the scored Top 10. MRR rewards ranking it closer to first. MTTC rewards finding it earlier in the conversation.

See `docs/SCORING_AND_JUDGING.md` and `REPORT.md` for the full measured story and limitations.

## Caveats

- No hosted inference API is used.
- The selected reranker needs prepared local model files.
- Sparse fallback is legal and useful, but it is not the selected neural result.
- Public sessions have been used for development; do not treat them as a fresh holdout.
- The private-like validation pack is authored engineering validation, not proof of private leaderboard performance.
- Keep generated artifacts, model files, catalogs, credentials, and private traces out of Git.
