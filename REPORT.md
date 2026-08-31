# Mercury — current lexical paging pipeline

The public release is **lexical retrieval with an adaptive shortlist and guarded
paging**, not the former neural D30/MiniLM pipeline. `agent.Agent` and
`starter.agent.Agent` export the same `mercury.lexical.Agent`; its selected
configuration is `DEFAULT_AGENT_CONFIG` in `mercury/lexical/config.py`.

The current source-bound public evaluation finds **200/200 targets** with a
**0.967414 TechnicalScore**, zero model tokens, zero agent errors and zero
fallbacks. These are consumed public development results, not private-test
predictions or measured shopping satisfaction.

## Implementation and repairs

The runtime parses source-aware preferences and corrections, combines bounded
FTS5/exact-constraint retrieval routes, guards category admission, ranks against
active constraints, plans a clarification, chooses shortlist width, and applies
paging inside the response transaction. It requires Python and SQLite FTS5, with
no third-party package or model required for the selected default.

This cleanup fixes the reviewed category drift, declined-question bypass and
rejection-paraphrase contamination. It also handles explicit replacement straps
misfiled under bags, consistent short-token/department normalization, closed-agent
lifecycle, legal clarification attributes, and decimal budgets in mixed rejection messages. Paging history is isolated by
session and cleared on reset, eviction, profile deletion and close. Failed turns
and identical retries cannot consume another page. Research adapters share the
selector but explicitly disable paging in their base to prevent double paging.

The README, setup, design, current demo/showcase, scoring guide and release
checklist now describe this same runtime. Historical configurations, results and
renderers are retained and labeled; they do not select the public pipeline.
See [design](docs/DESIGN.md), [setup](docs/SETUP.md), and
[research index](docs/RESEARCH_INDEX.md).

## Current measurement — 31 August 2026

| Measure | Current source |
|---|---:|
| Public sessions | 200 |
| HitRate@10 | 1.000000 (200/200) |
| MRR | 0.965048 |
| MTTC | 2.105 |
| Efficiency | 0.889500 |
| TechnicalScore | 0.967414 |
| Prompt / completion tokens | 0 / 0 |
| Agent errors / fallback turns | 0 / 0 |
| Agent construction | 9.963 seconds |
| Response p50 / p95 | 0.111 / 0.397 seconds |
| Peak process RSS | 1,266,384,896 bytes |

The run used Python 3.13.5 with site-packages disabled and socket connections
denied. RSS includes the evaluator, catalog, agent and instrumentation; it is
not an isolated per-agent allocation. Timings are local observations, not host
limits or latency guarantees. Paid model/API inference cost is zero; local
hardware, electricity and storage are not free.

[Current-results.json](docs/current-results.json) records configuration, source
hashes, catalog/dataset/evaluator hashes and report SHA-256. The complete report
is local at `output/lexical-cleanup/public-release/report.json`; raw traces
are ignored, not published. Source changes invalidate its current-build claim.
Reproduce with:

```bash
python -S -m experiments.submission_evaluate --output output/new-public-verification
python -S -m demo.showcase --output output/new-current-demo
```

Use an unused destination. The demo accepts an optional verified aggregate only
with both its report path and expected SHA-256; see [demo guide](docs/DEMO_SCRIPT.md).

## Improvement history and evidence boundaries

This section intentionally retains earlier results to explain repairs and trade-offs; the current result is the table above.

The first cleanup measurement exposed an overly strict category normalization
bug: 190/200 hits and 0.919164 TechnicalScore. Query-side short tokens such as the
“T” in “T-Shirts” were retained while product-side category tokens discarded
them; partial combined department headings also mismatched. General authored
regressions cover these cases, and normalization was corrected before the final
measurement. A final decimal-budget parser repair was checked before refreshing the source-bound release measurement. The initial report is preserved at
`output/lexical-cleanup/public-verified/report.json`; it is not hidden or presented
as the final source's score. No ranking thresholds were tuned from that run.

Earlier frozen lexical paging scored 0.968064 on public data. The current result
is 0.000650 lower with the same 200/200 hit count; this correctness cleanup is not
a claim of a score improvement. The historical robustness-v1 final paging result
was 77/80 and 0.899406, but it belongs to its frozen pre-cleanup source. That final
set was already consumed and was **not reopened or used to tune this repair**.
See [dataset exposure status](docs/DATASET_STATUS.md). Further claims about unseen
performance require genuinely new, independently reserved evaluation.

## Verification and remaining limits

All 1,164 repository tests pass. The 130 dependency-free runtime/integration
checks pass with site-packages disabled; Ruff and diff whitespace checks pass.
A separate minimal package containing only public entry points and lexical
helpers runs with no evaluator/research imports, models or socket access. The
current full-catalog demo records real category-stable corrections and paging,
and the public agent matches the experimental paging adapter on authored inputs.
See [verification details](docs/PIPELINE_VERIFICATION.md).

This is not a proof that all inputs are bug-free. The finite English parser,
incomplete taxonomy, uncertain/contradictory catalog descriptions and identical
products remain limitations. Constraint guards conservatively demote known
violations; they do not certify physical product properties or universally
suppress all contradictory records. Paging preserves the base violation quota,
and can repeat after reset, ranking changes or compatible-tier exhaustion.
Concurrent calls require external serialization. Real-user benefits and final
organizer-host resource compliance remain unverified.

Public repository visibility, team/contribution details, video production/upload
and Devpost submission are separate external gates. See [release checklist](docs/RELEASE_CHECKLIST.md).
Historical improvement comparisons are listed in [research and release boundaries](docs/RESEARCH_INDEX.md); redundant old release-guide snapshots now point to current documentation.
