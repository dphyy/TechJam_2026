# Setup for the public lexical pipeline

Run from the repository root with Python 3.10+ and SQLite FTS5. Current local
verification uses Python 3.13.5 on macOS. No third-party package, model download,
provider key, neural asset, or JSON configuration is required by `agent.Agent`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE probe USING fts5(text)'); c.close()"
```

Obtain `catalog.jsonl.gz` and checksums from the organizer's release described in
[data/README.md](../data/README.md). Verify the archive against those checksums
before decompression; keep the original archive.

```bash
gzip -dk data/catalog.jsonl.gz
shasum -a 256 data/catalog.jsonl
```

Expected decompressed SHA-256:
`da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67` (50,000 rows).
The public development file's SHA-256 is
`857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`.
Do not change evaluator, catalog or test labels. Follow [data attribution](../DATA_ATTRIBUTION.md).

```bash
python -m demo.submission --output output/current-demo
python -m experiments.submission_evaluate --output output/public-verified
python -S -m unittest tests.test_lexical_paging tests.test_lexical_state \
  tests.test_guarded_paging_evaluate tests.test_submission_demo tests.test_submission_evaluate
```

The public runtime needs `agent.py`, `starter/agent.py`, the `mercury/lexical/`
package and package initializers, plus the supplied catalog. Shipping the full
`mercury/` directory is simplest. Evaluator/demo commands additionally need their
repo modules. Runtime construction does not import evaluator, demo or tests.
It builds an in-memory index, or validates and uses a compatible local persisted
catalog index when one is present. No index artifact is mandatory.

For linting, install `requirements-dev.txt`. For the full historical research
test suite and optional neural experiments, install `requirements-research.txt`
and follow [model documentation](MODELS.md) only if using those experiments.
The archived requirements lock was tested on a different Python/platform; it is
not a universal cross-platform installation guarantee. Installation may need the
network; default lexical execution does not. `configs/selected.json` only governs
the old neural agent. See [README](../README.md) for the API and current commands.
