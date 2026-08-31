# Mercury

**Conversational product search that remembers changing intent and explores beyond the first shortlist.**

`Python` · `SQLite FTS5` · `Offline` · `Lexical search + guarded paging`

[Overview](#project-overview) · [Setup](#setup-and-installation) · [Results](#reproducing-the-results) · [Reflection](#limitations-and-future-improvements) · [Team](#team-member-contributions)

---

## Project overview

Mercury searches a local product catalog through a ten-turn shopping conversation.
The public pipeline is **SQLite FTS5 lexical retrieval → constraint-aware ranking
→ adaptive shortlist → guarded paging**. It remembers explicit corrections and
exclusions, asks clarifying questions, and returns real catalog IDs with evidence
receipts. No model, embedding index, hosted API, or paid inference is required.

`agent.Agent`, `starter.agent.Agent`, and `mercury.lexical.Agent` are the same class.
The default is defined in `mercury/lexical/config.py`, not `configs/selected.json`.
The latter belongs to the former neural pipeline, retained for research only.

## Setup and installation

Use Python 3.10+ with SQLite FTS5. The current checks use CPython 3.13.5 on macOS;
other hosts still need verification. The default runtime has no required Python
packages. Clone or download this repository, open a terminal in its root directory,
and create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain the organizer's catalog using [the data instructions](data/README.md),
verify the organizer's archive checksum, and decompress it to `data/catalog.jsonl`.
The 50,000-row catalog used here has SHA-256
`da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.
The catalog is not bundled in source control. After placing the verified archive
at `data/catalog.jsonl.gz`, decompress it and check the resulting file:

```bash
gzip -dk data/catalog.jsonl.gz
shasum -a 256 data/catalog.jsonl
```

The checksum must match the value above. The public development sessions are
already included at `data/public_set.jsonl`. See [detailed setup](docs/SETUP.md)
for platform checks and optional research dependencies.

### Try the Agent API

```python
from agent import Agent

agent = Agent("data/catalog.jsonl")
try:
    agent.reset("shopper-1", {})
    print(agent.respond("shopper-1", "I'm looking for bags. A key requirement is: canvas.", 1, 10))
    print(agent.respond("shopper-1", "Those options aren't right.", 2, 10))
    print(agent.respond("shopper-1", "Correction: make that leather.", 3, 10))
finally:
    agent.close()
```

Call `reset` first. Turn numbers must advance from 1 through 10; an identical
retry of the latest request returns the cached response without advancing paging.
The official evaluator requests `top_k=10`; the implementation also accepts
1–10 for local callers and may return a shorter slate. State commits only after
search, planning, selection, and response validation succeed.

## Pipeline behavior

1. Parse messages into source-aware category, preference, correction, exclusion,
   and no-preference evidence. Rejection of a displayed slate is not a product
   preference; separately stated requirements in the same message are retained.
2. Retrieve through broad, phrase, category, and exact-constraint lexical routes.
   Exclude known incompatible taxonomy before ranking. Missing taxonomy remains
   unknown. Explicit replacement-strap titles do not satisfy a whole-bag request.
3. Rank with contradiction guards, exact constraint matches, category specificity,
   catalog field evidence, and bounded quality/profile tiebreaks. Unknown metadata
   is not proof that a requirement is met. Ranking scores are not probabilities.
4. Choose a useful clarification and adaptive output width. Do not reopen a
   declined `other` question; stop asking questions on turn 10.
5. Page within the ranked context of at most 100 candidates when active semantics
   and the top-ten candidate membership are unchanged. Prefer unseen candidates,
   preserve the current shortlist width and known-violation quota, and reuse
   compatible seen candidates when that tier is exhausted. A semantic change or
   explicit correction resets exposure and replays the current best matches.

Paging does not guarantee zero repetition. Resets, changed rankings, and an
exhausted compatible pool can repeat products. It also does not turn incomplete
or contradictory catalog metadata into verified product facts. See [design and
limitations](docs/DESIGN.md).

## Reproducing the results

After completing setup, run the unchanged official evaluator through the
instrumented public-agent runner. This is the command used for the reported
pipeline, with site-packages disabled:

```bash
python -S -m experiments.submission_evaluate \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output output/public-reproduction
```

Use a new output directory for each run. The runner creates `registration.json`,
`report.json`, and `traces.json`, recording source/configuration/data hashes,
legality checks, timings, and individual session outcomes. Compare the aggregate
fields in `report.json` with these results from the verified source:

| Public development metric | Result |
|---|---:|
| Targets recovered | **200 / 200** |
| TechnicalScore | **0.967414** |
| Mean reciprocal rank (MRR) | 0.965048 |
| Mean turns to completion (MTTC) | 2.105 |
| Model tokens | 0 |
| Agent errors / fallback turns | 0 / 0 |

These are consumed development results, not private-test predictions. Runtime
timings vary by machine. Exact score reproduction requires the same source,
configuration, catalog, and dataset; the recorded hashes are in
[current-results.json](docs/current-results.json).

For an uninstrumented run through the official CLI:

```bash
python -m evaluator.local_evaluator --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl --output output/public-results.json
```

### Replay the conversation

```bash
python -m demo.submission --output output/current-demo
# Equivalent public-agent showcase:
python -m demo.showcase --output output/current-showcase
```

Use a new output directory for each recorded run. The demo writes `index.html`,
`transcript.txt`, and sanitized `evidence.json`, showing real corrections and
paging. Aggregate scores can be attached only with a matching source-bound report
and its SHA-256; arbitrary or historical result JSON is not accepted. These
commands do not upload, publish, or submit anything.

Current verification and score status live in [the report](REPORT.md),
[verification notes](docs/PIPELINE_VERIFICATION.md), and
[current-results.json](docs/current-results.json). Earlier high scores do **not**
automatically apply to changed code. Public and previously opened synthetic sets
are consumed development evidence. Robustness-v1's final set was already consumed;
this repair does not retune against or reopen it. Follow [dataset status](docs/DATASET_STATUS.md).

## Checks and repository map

```bash
# Dependency-free runtime/integration regressions, including paging:
python -S -m unittest tests.test_lexical_paging tests.test_lexical_state \
  tests.test_guarded_paging_evaluate tests.test_submission_demo tests.test_submission_evaluate

# Optional lint tool:
python -m pip install -r requirements-dev.txt
python -m ruff check .

# Full historical + current suite in the optional research environment:
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -q
```

| Location | Role |
|---|---|
| `agent.py`, `starter/agent.py` | Public submission entry points |
| `mercury/lexical/` | Current parser, retrieval, ranking, planner, paging, diagnostics |
| `demo/submission.py`, `demo/showcase.py` | Current public-agent recording and CLI alias |
| `experiments/submission_evaluate.py` | Current source-bound evaluation |
| `experiments/guarded_paging_evaluate.py` | Explicit paging-on/off ablation using the shared selector |
| `evaluator/`, `data/public_set.jsonl` | Unchanged organizer scoring and public development data |
| `mercury/agent.py`, `mercury/fusion/`, `configs/`, other experiments | Historical or optional research; not the public default |
| `docs/history/`, [research index](docs/RESEARCH_INDEX.md) | Historical comparison evidence and redirects from superseded guides |

Current-facing guides retain only the newest result. Older scores are kept only
in explicitly labeled improvement, regression and experiment records; the
[documentation audit](docs/DOCUMENTATION_STATUS.md) lists those exceptions.

## Limitations and future improvements

Our main lesson is that recovering an exact product in a simulator and helping
a real shopper are related but different goals. Explicit state and guarded
paging make the behavior inspectable, but a high development score does not prove
that every recommendation is relevant or that shoppers save time.

This is an English rule-based research backend, not general language understanding
or a production commerce service. Catalog errors, unseen taxonomy synonyms,
complex negation, and ranking ambiguity remain limitations. Missing metadata is
not proof of compliance, and paging can repeat products after resets or pool
exhaustion. Concurrent use of a single agent requires external serialization.

Given more time, we would prioritize:

- **Independent evaluation:** reserve genuinely new test sets and run real-user
  studies measuring relevance, satisfaction, and time saved, without retuning
  against consumed holdouts.
- **Better intent and catalog understanding:** expand taxonomy/synonym coverage
  and correction/negation handling, with authored regressions and explicit
  treatment of missing or contradictory product information.
- **Deployment readiness:** verify organizer-host resource limits, profile
  startup and memory costs, and add safe concurrent session handling before
  exposing the backend as a service.

Public video, repository visibility, and competition submission remain separate
[release gates](docs/RELEASE_CHECKLIST.md). Dataset provenance and use constraints
are documented in [data attribution](DATA_ATTRIBUTION.md).

## Team member contributions

| Team member | Contribution role |
|---|---|
| **Brandon** | AI Ranking Optimisation Engineer |
| **Danvern** | AI Integration & Optimisation Engineer |
| **Kingold** | AI Feature Development Engineer |
| **Saai** | AI Pipeline Design Engineer |
| **Zhi Kai** | AI Experimentation & Evaluation Engineer |
