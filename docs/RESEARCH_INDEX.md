# Research and release boundaries

The active runtime is documented in [README](../README.md) and [design](DESIGN.md).
Only `agent.Agent` / `mercury.lexical.Agent` with `DEFAULT_AGENT_CONFIG` defines the
public default: lexical retrieval, adaptive shortlist, guarded paging.

Other reports/protocols in this directory preserve named experiments and the
source/configuration receipts that justified their conclusions. A historical
statement such as “selected,” “current,” “sealed,” or “promoted” applies at that
record's date; it does not override the current runtime or consumption ledger.
See [dataset exposure status](DATASET_STATUS.md) before calling any set held out.

| Material | Current interpretation |
|---|---|
| `history/*-before-lexical-paging.md` | Redirects only; duplicate outdated guide bodies have been removed |
| `history/neural-public-results.json` | Historical neural comparison artifact; never a current lexical score |
| Cycle, refinement, review-prior, D30/MiniLM and neural reports | Historical neural/retrieval experiments; run their recorded source/config |
| `ADAPTIVE_GUARDED_PAGING_PROTOCOL.md` | Original opt-in paging study; selection logic is now shared with production |
| `experiments/presentation_evaluate.py` | Isolated unpaged-base presentation policies, not a hidden public-agent selector |
| `experiments/guarded_paging_evaluate.py` | Paging-on/off ablation of current shared selector with an unpaged base |
| `experiments/submission_evaluate.py` | Source-bound evaluation of the current public entry point |
| `configs/selected.json`, `mercury.agent.Agent` | Former neural release, retained for reproducibility |
| `mercury/fusion/` | Separate fusion research; explicitly unpaged controls |
| `demo/legacy_showcase.py`, other non-submission demos | Historical research recordings |

Recorded historical code revisions are necessary to reproduce old scores exactly. Extracting
common runtime logic changes source receipts and invalidates old reports as proof
of current behavior; it does not erase the original results. Runtime does not
read any report, target label, dataset split, or evaluation outcome.

Local `output/`, `runs/`, `artifacts/`, downloaded catalog/model files and raw
traces are ignored. Aggregate disclosure does not authorize redistribution of
catalog data, private synthetic examples or conversation traces.

The [documentation status and exception list](DOCUMENTATION_STATUS.md) enumerates
every retained comparison/protocol, explains why old numbers remain there, and
identifies which pages contain only current guidance. Failed experiments remain
visible alongside improvements; preserving an unfavorable result prevents a
misleading history of only successful changes.
