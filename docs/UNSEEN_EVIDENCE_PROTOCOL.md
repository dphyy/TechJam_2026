# Unseen-Target Evidence Protocol

Registered 28 August 2026 before running any roadmap candidate on these rows.

## Purpose

The released 200-session public set is consumed development evidence. Roadmap
promotion therefore requires catalog targets that were not present in that set.
This protocol creates deterministic evaluator-compatible sessions before agent
inference and keeps a final split sealed.

This is synthetic unseen-target engineering evidence, not organizer-private or
real-user evidence. The official evaluator still derives intent cards and dialogue
from frozen product metadata.

## Preparation

```bash
python -m experiments.unseen_prepare \
  --catalog data/catalog.jsonl \
  --exclude data/public_set.jsonl \
  --output artifacts/unseen-v1 \
  --development-count 80 \
  --final-count 40 \
  --seed 20260828
```

The command is create-only. It excludes all 200 public targets, selects unique
catalog products without running Mercury, assigns unique synthetic user groups,
balances five coarse product families, and preserves the official 40/40/15/5
scenario mix. Development and final targets/users are mutually disjoint.

Frozen content hashes:

```text
development.jsonl  3cba45ace6494f28e1fdc707f57a335b9f028314b9a3f773f666cb50992a7fae
final-sealed.jsonl  54938a5aa0a1075f2d7013be79e041f909cba61f0722499b769b6be8315b9d64
```

The generated rows and manifest remain ignored local artifacts. The preparation
source, tests, command, seed, input checksums, and expected output hashes are
committed so another team member can reproduce them exactly.

## Use and Consumption

1. Use `development.jsonl` for threshold fitting and primary ablations.
2. Keep `final-sealed.jsonl` unopened by evaluation until one candidate is frozen
   after correctness, latency, and development gates.
3. Record source, config, model, input hashes, latency, memory, tokens, fallbacks,
   and per-scenario results for every run.
4. A failed or interrupted final run consumes the final split for that source and
   configuration; do not repair and relabel it as unseen.
5. Do not use the public set to tune new roadmap thresholds. It may be replayed
   later only as labeled descriptive compatibility evidence.

## Failure Taxonomy

Every candidate report must distinguish:

- state/intent representation failure;
- retrieval miss;
- rerank-admission miss;
- ranking/order miss;
- question/dialogue decision failure;
- constraint/evidence failure;
- runtime/model fallback or API error.

Categories may overlap, but the report must identify its classification rules and
retain raw target-independent runtime diagnostics. Ground truth is used only by
the offline report, never passed to the Agent.

The first development-only ablation is recorded in
[the roadmap implementation results](ROADMAP_IMPLEMENTATION_RESULTS.md). No arm
passed promotion, so the sealed final split remains unconsumed.
