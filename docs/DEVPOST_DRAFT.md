# Submission draft: Mercury

Draft only. Public repository URL, video URL, team details and submission status
must be completed from verified facts; no upload or submission is implied.

## Problem and approach

Shopping intent changes during a conversation. Mercury is a local backend that
keeps explicit requirements, exclusions, corrections and no-preference boundaries
inspectable while searching 50,000 catalog products. It combines lexical retrieval,
constraint-aware ranking, an adaptive shortlist and guarded paging to make progress
when the user has no further useful details.

## Current implementation

The public Python Agent uses SQLite FTS5 and the standard library. It does not
require a neural model or hosted API. Category guards prevent obvious unrelated
products from winning on shared feature words. The planner respects declined
questions. Paging prefers unseen candidates when semantics and the leading set
are stable; corrections reset exploration. Retries and failures cannot accidentally
advance the page. Actual output and evidence receipts can be replayed offline.

## Evidence and limitations

The latest verified source recovers **200/200 public targets**, with
**TechnicalScore 0.967414**, MRR 0.965048 and MTTC 2.105, with zero model tokens,
agent errors or fallbacks. These are consumed development results; source and
configuration receipts are in [current-results.json](current-results.json) and
[REPORT](../REPORT.md). Historical paging studies and
robustness results describe their frozen versions, not all later fixes. No
organizer-private or real-user result is claimed. The consumed robustness-v1 final
set was not used to tune this cleanup.

Catalog ambiguity, incomplete taxonomy and finite English parsing remain limits.
Paging may repeat after a reset, ranking change or compatible-pool exhaustion.
The demo is an authored conversation with actual API output, not a scored target
or proof of universal correctness. See [demo guide](DEMO_SCRIPT.md).

## Tools, data and release fields

- Python, SQLite FTS5, unittest; optional Ruff for linting.
- Organizer text/metadata catalog derived from Amazon Reviews 2023; see
  [data attribution](../DATA_ATTRIBUTION.md). No model weights in the default runtime.
- API: the supplied local Python Agent/evaluator contract; no external runtime API.
- Public repository: **pending verified URL/visibility**.
- Public video: **pending recording/review/upload**.
- Team members and roles: **provided**, recorded in the [README](../README.md#team-member-contributions).
- Dated contribution/build-window record: **still to be completed and verified**.

Follow [release checklist](RELEASE_CHECKLIST.md); replace placeholders before submission.
