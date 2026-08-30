# Private-Like Engineering Validation

This pack is for judge-visible engineering confidence, not leaderboard tuning or private-test prediction.

It complements the official public simulator with small authored catalog conversations that target known risk areas:

- vague ordinary queries
- intent correction
- explicit negation
- no-preference replies
- alternatives
- body/component evidence
- accessory versus primary object confusion
- sparse metadata
- negative feedback after bad recommendations

The fixture lives at `data/private_like_capabilities.json` and reuses the strict capability-result schema from `experiments.cycle2_capabilities`. Each case has an isolated mini catalog, one to three visible user turns, and explicit assertions over rankings, active preferences, unknown evidence, and hard-exclusion behavior.

Run a config with:

```bash
.venv/bin/python -m experiments.private_like_validate \
  --config configs/selected.json \
  --output runs/private-like-selected
```

A passing case means the configured runtime satisfied the authored assertion under the fixture catalog. It does not prove broad semantic correctness, real-user usefulness, or organizer-private performance. Failed cases should stay visible until a new registered improvement cycle fixes them and compares against current Mercury on both public and private-like validation.

Use the comparison harness for official-style target recovery:

```bash
.venv/bin/python -m experiments.evaluate_suite \
  --output runs/evaluation-suite
```

That report compares the original baseline, sparse Mercury fallback, selected Mercury, and any `--candidate NAME=PATH` configs on the same catalog/dataset.
