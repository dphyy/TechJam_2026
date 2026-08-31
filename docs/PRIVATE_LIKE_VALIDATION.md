# Authored engineering validation

For the current lexical search with guarded paging, run the authored regressions
and current API/demo checks from the repository root:

```bash
python -S -m unittest tests.test_lexical_paging tests.test_lexical_state \
  tests.test_guarded_paging_evaluate tests.test_submission_demo tests.test_submission_evaluate
```

The latest recorded verification contains 130 passing checks in that command and
1,164 passing tests in the complete research environment. See
[verification](PIPELINE_VERIFICATION.md) for source and measurement boundaries.
These are invented-catalog correctness and integration checks, not organizer-private
scores or proof of real-shopper usefulness. The current full-catalog demonstration
is `python -m demo.submission --output output/current-demo`; use a new output path.

`data/private_like_capabilities.json`, `experiments.private_like_validate`, and
`experiments.evaluate_suite` belong to the older configurable neural research
pipeline. They do not validate the public lexical default merely because their
command accepts `configs/selected.json`. Prior capability results are preserved
in the [historical comparison records](RESEARCH_INDEX.md). Use their recorded
source and configuration when reproducing those experiments.

Do not treat a passing fixture, a development target score, or the filename
“private-like” as independent private-test evidence. See [dataset status](DATASET_STATUS.md).
