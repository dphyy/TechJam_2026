# Current release checklist

Current public pipeline: **lexical search, adaptive shortlist, guarded paging**.
`agent.Agent` and `starter.agent.Agent` export `mercury.lexical.Agent` with
`DEFAULT_AGENT_CONFIG`. No neural configuration, model or hosted API is required.
Only this checklist describes the current release. Historical comparisons are indexed in [research and release boundaries](RESEARCH_INDEX.md).

## Local implementation and evidence

- [x] Integrate the shared guarded-paging selector into the public agent transaction.
- [x] Keep research presentation/fusion controls explicitly unpaged; prevent double paging.
- [x] Guard category admission, including broad shoe requests and obvious replacement straps.
- [x] Honor declined open questions on every planner path and handle rejection paraphrases.
- [x] Cover reset, retry, failed-turn rollback, session eviction, close and profile deletion.
- [x] Use the public pipeline in the current demo/showcase and record actual paging receipts.
- [x] Replace stale setup/current-release claims; separate minimal runtime from optional research dependencies.
- [x] Preserve official evaluator, catalog and public dataset; no final-set tuning in this cleanup.
- [x] Keep outputs, traces, private data and model/index assets ignored.

See [verification notes](PIPELINE_VERIFICATION.md) for actual checks and
[current-results.json](current-results.json) for the source-bound measurement
status. A checkmark above does not imply organizer-host or real-user validation.

## Remaining external gates

- [ ] Verify intended public source location, data/code redistribution rights and current visibility.
- [ ] Verify target Python/SQLite availability and organizer CPU, memory, disk, startup and turn limits.
- [ ] Confirm how the supplied track brief maps to the applicable judging stage.
- [x] Record the five owner-provided team members and contribution roles in the README.
- [ ] Complete the dated contribution record and verify significant competition-period updates.
- [ ] Produce and review the required public demo video, then approve/upload it separately.
- [ ] Fill approved repository/video URLs and final source-specific metrics in submission material.
- [ ] Complete the Devpost submission before the verified deadline; retain confirmation.

The [official rules](https://tiktoktechjam2026.devpost.com/rules), checked 31 August
2026, display 1 September 2026 at noon SGT as the deadline and require a public
code repository with README and a project description. Follow additional track
and [event deliverables](https://tiktoktechjam2026.devpost.com/) as applicable.
Local cleanup or a Git push does not authorize video upload, outreach, paid
compute, public data release or competition submission.

Preserve history and explicitly stage intended files when committing. Never
include downloaded catalog data, raw evaluation traces, credentials or unrelated
working files. Verify the remote commit after any separately requested push.
