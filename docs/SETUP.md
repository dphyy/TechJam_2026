# Backend setup

Run commands from the repository root. The landing README provides the short path; this file contains the detailed reproducibility instructions.

## Python and dependencies

The tested environment is CPython 3.12.12 on macOS arm64. Python must include SQLite FTS5. Other operating systems, architectures and organizer hardware require their own verification.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The single dependency manifest includes runtime, local neural reranking, experiment tooling and developer checks. Installation uses the network but does not acquire model weights. A completely air-gapped installation needs preinstalled dependencies or a matching-platform wheelhouse; a fresh wheelhouse installation has not been verified.

## Organizer catalog

Obtain `catalog.jsonl.gz` and the published checksums from the organizer using [the data instructions](../data/README.md). Verify the archive before decompression. Place it in `data/` and keep the original archive:

```bash
gzip -dk data/catalog.jsonl.gz
shasum -a 256 data/catalog.jsonl
```

The frozen 50,000-product decompressed catalog has SHA-256 `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`. Do not alter catalog rows or evaluation labels. Catalogs, raw traces, model files and virtual environments stay outside source control.

## Local reranker

Acquire the pinned reranker explicitly during preparation, then verify it without downloading:

```bash
python -m experiments.prepare_models --model reranker --download
python -m experiments.prepare_models --model reranker
```

Required assets live in `artifacts/models/reranker/`. The selected model is `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, Apache-2.0. Preserve its notices and file-hash manifest. Model loading is local-only, safetensors-only, disables remote code and requires no provider credentials. See [model details](MODELS.md).

The selected configuration uses grouped explicit alternatives, four CPU threads,
120 retained candidates, a 30-candidate reranking prefix, and unchanged-rank
slate paging from the first repeated ranking with a page-1 reset on detected
intent overrides. The original grouped-alternatives selection receipt is
historical; the paging promotion and current settings are recorded in
[the merged refinement results](PIPELINE_REFINEMENT_RESULTS.md). No dense index
or contrast sidecar is required. Missing or invalid optional assets trigger a
recorded sparse fallback; that is a degraded mode, not reproduction of the neural
result. Do not silently report fallback measurements as neural measurements.

## Interface and official harness

Package [agent.py](../agent.py), its `mercury/` helper modules and [configs/selected.json](../configs/selected.json) together. The entrypoint resolves model paths relative to its own directory. Omitting the selected configuration activates different defaults and does not reproduce the selected build.

```python
from agent import Agent

agent = Agent("data/catalog.jsonl")
try:
    agent.reset("example", {})
    print(agent.respond("example", "I need a canvas bag.", 1, 10))
    print(agent.respond("example", "Actually, blue please.", 2, 10))
finally:
    agent.close()
```

Use `reset` before the first response. Turns range from 1 to 10, and each response contains at most ten unique real catalog IDs. The official contract is [agent_api_contract.json](agent_api_contract.json). No target, scenario label or evaluation outcome is supplied to the agent.

Run the unchanged organizer harness on the released public set:

```bash
python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results.json
```

The released 200 sessions have been used for development and are not an independent holdout. For a named, instrumented development run, choose an unused output name:

```bash
python -m experiments.run --name local-selected-repro --config configs/selected.json --dataset data/public_set.jsonl
```

Keep benchmarks serial and retain their manifests. The runner records source/configuration/data hashes, package versions, actual tokens, fallbacks, cold start, response latency and peak process memory. It does not charge or use a hosted inference API. Local compute still consumes time, electricity and storage.

## Tests and judge-facing replay

```bash
python -m unittest discover -s tests -q
python -m ruff check .
python -m demo.showcase --results docs/current-results.json --output artifacts/judge-showcase
python -m demo.alternatives --catalog data/catalog.jsonl --selected-mode grouped --output artifacts/local-alternatives-replay
```

The showcase is the recommended judge walkthrough. It executes the real selected agent and writes a portable `index.html` plus machine-readable `evidence.json`; use an unused output directory. The alternatives replay is retained as historical Cycle 2 evidence and executes all three fixed controls. Neither command uploads or submits anything. See [the demo guide](DEMO_SCRIPT.md).

## Completed comparison and validation boundary

The alternatives comparison is complete under [its registered protocol](CYCLE2_ALTERNATIVES_PROTOCOL.md). [The final report](CYCLE2_RESULTS.md) and [aggregate evidence](cycle2-summary.json) distinguish developer correctness, target recovery, capability failures and resources. Grouped passed the release gates but did not improve target scores or locked capability passes. Its historical fixed-slate public score is 0.786724; the current realistic-shopping implementation scores 0.844994 on that consumed set. The latter is an implementation check, not a fresh private-performance estimate.

Reproduce a control on released development data with a fresh name:

```bash
python -m experiments.run --name local-frozen-repro --config configs/cycle2_frozen.json --dataset data/public_set.jsonl
python -m experiments.run --name local-parse-repro --config configs/cycle2_parse.json --dataset data/public_set.jsonl
python -m experiments.run --name local-grouped-repro --config configs/cycle2_grouped.json --dataset data/public_set.jsonl
```

The new target/capability inputs and raw artifacts remain local, not in Git; reproducing their exact results requires those preserved packs and manifests. All six registered validation jobs are already consumed. The freeze verifier deliberately rejects the promoted checkout because `configs/selected.json` now contains grouped rather than the pre-selection OFF baseline. The source snapshot, pre-promotion verification and selection receipt explain that intentional difference. Do not weaken the verifier, remove consumption markers, reopen validation to test installation, or claim a new output directory is a fresh holdout.
