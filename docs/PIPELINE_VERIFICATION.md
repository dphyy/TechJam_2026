# Lexical paging cleanup verification — 31 August 2026

Scope: the public lexical pipeline and its paging, category, dialogue, demo and
release-documentation consistency. The official evaluator, supplied catalog and
public dataset were not changed. The consumed robustness-v1 final set was not
reopened. Private outputs remain ignored under `output/lexical-cleanup/`.

## Repaired failure classes

| Failure | Repair and evidence |
|---|---|
| Shoes/caps outrank bags on shared canvas/strap text | Category admission before lexical tiers; invented mixed-category cases and real 50k-catalog replay |
| Broad shoes disappear during category normalization | Preserve single-category department words; normalize combined department prefixes symmetrically |
| Exact taxonomy rejected for short tokens | Same short-token normalization on both sides; T-Shirts, ID Cases and No Show Socks regressions |
| Replacement straps treated as whole bags | Narrow explicit replacement-part title guard; bags with included spare straps retained |
| Declined `other` question reopened through fallback | Refusal and repeat cap checked on every route; balanced 16-shirt case across turns |
| Rejection paraphrase becomes positive preference | Shared conservative feedback parser; rejection-only signatures unchanged, mixed new needs retained |
| Paging opt-in only / duplicate paging risk | Selector in public agent; ablation base explicitly unpaged; six-turn public/adapter response parity |
| Failed turn or retry advances page | Paging staged with dialogue; failure injection proves rollback, retry preserves cached output |
| Rejection parsing splits decimal budgets | Preserve decimal points and thousands separators; authored 10.50 and 1,250.75 budget regressions |
| Session/profile lifecycle leaves exposure behind | Reset, eviction, profile deletion and close clean paging state; closed calls rejected |
| Stale demo/metrics/configuration claims | Public showcase alias, verified paging receipts, strict metric/source binding, historical records labeled |

## Checks completed

- `python -m unittest discover -s tests`: **1,164 passing tests** in the existing
  research environment. This includes old research functionality and new runtime regressions.
- `python -S -m unittest tests.test_lexical_paging tests.test_lexical_state
  tests.test_guarded_paging_evaluate tests.test_submission_demo
  tests.test_submission_evaluate`: **130 passing checks**, no site-packages.
- `python -m ruff check .` and `git diff --check`: pass.
- Minimal copied package: public entry points plus `mercury/lexical` and package
  initializers only; authored catalog; site-packages disabled; socket connections
  blocked; paging and retry work; no evaluator, experiments, tests, demo, NumPy,
  Torch or sentence-transformers module is loaded.
- Public agent and explicitly unpaged-base experimental adapter: identical actual
  responses across six authored preference/rejection/correction turns.
- Full 50,000-product five-turn public showcase: healthy, category-stable, excludes
  the observed replacement-strap failure, and records paging on both rejection turns.
  The final source-bound aggregate is attached by verified report checksum.
- Active README/setup/design/report/rubric/demo/release links checked; historical
  source/config records remain explicitly separated from current selection.

## Repair comparison history

Earlier measurements in this section are intentional regression/repair comparisons, not alternative current scores.

The first public verification measured **190/200, 0.919164** and exposed the
normalization defect. Its receipt is retained at
`output/lexical-cleanup/public-verified/report.json`. After a general normalization
repair and authored regressions, the final run measured **200/200, 0.967414**.
A subsequent decimal-budget repair was followed by a final source-bound rerun with the same aggregate score; the intermediate run is also retained at `output/lexical-cleanup/public-verified-final/report.json`. There was no parameter search or final-set tuning. See [REPORT](../REPORT.md) and
[current-results.json](current-results.json) for metrics and exact source identity.

Final report SHA-256:
`a3019c7242e9f23ebf0f118790745ce3cdb40783fac1be11d69b9c066e8379f2`.
Combined source-receipt SHA-256:
`579de789778d7f78bdb4a2ba90b6f26426255c6702fbf01070d9fadc1dad7148`.

A local passing suite does not establish unrestricted language correctness,
organizer-private accuracy, real-user impact or production concurrency safety.
Remaining limits and external release gates are explicitly listed in
[design](DESIGN.md) and [release checklist](RELEASE_CHECKLIST.md). No public upload,
competition submission, or Git push is performed by these checks.
