# Research configurations

The public `agent.Agent` uses `mercury.lexical.config.DEFAULT_AGENT_CONFIG`:
deterministic lexical search, adaptive shortlist, and guarded paging. It does
not read `selected.json` or any other JSON file in this directory.

These files retain named neural/fusion/retrieval experiments for reproducibility.
In particular, `selected.json` is the former neural release configuration, not
an alias for the current public pipeline. Use the matching experiment runner
and source revision recorded in each historical report. Changing this directory
does not promote an experiment to the public entry point.

See [the current README](../README.md), [design](../docs/DESIGN.md), and
[research index](../docs/RESEARCH_INDEX.md).

Any old scores or selection rules belong in the explicitly labeled comparison
reports, not in this configuration guide. The newest measured result is in
[current-results.json](../docs/current-results.json); the complete exception list
is in [documentation status](../docs/DOCUMENTATION_STATUS.md).
