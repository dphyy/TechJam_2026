# Current maintenance status

The public pipeline is **lexical search → adaptive shortlist → guarded paging**.
The runtime cleanup, source-bound public evaluation and README/team updates are
complete. Current behavior, results and verification are documented in
[README](README.md), [REPORT](REPORT.md), and
[verification](docs/PIPELINE_VERIFICATION.md).

Use [the release checklist](docs/RELEASE_CHECKLIST.md) for remaining external
publication, video, dated-contribution and submission work. Completion of local
checks is not evidence of external submission.

Do not retune against the consumed robustness-v1 final set or modify the official
evaluator/catalog/labels. Runtime changes require a new source-bound measurement;
public regression results remain descriptive development evidence.

Historical improvement plans and decisions are indexed in
[research and release boundaries](docs/RESEARCH_INDEX.md). They do not select the
current configuration or constitute pending implementation work. The latest
[documentation audit](docs/DOCUMENTATION_STATUS.md) identifies retained historical
exceptions and superseded guide links.
