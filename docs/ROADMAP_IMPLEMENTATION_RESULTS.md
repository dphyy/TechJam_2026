> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Robustness Roadmap Implementation Results

Recorded 28 August 2026 after implementing Phases 9-14 from
[the pipeline plan](../plan.md). All new runtime behavior is independently gated;
`configs/selected.json` remains the 30-candidate release.

Postscript, 30 August: after merging main, unchanged-rank slate paging passed a
new target/user-disjoint development and final comparison and is now enabled in
the selected D30 release. Intent routing and every Phase 9-14 behavior remain
disabled. See [the merged refinement results](PIPELINE_REFINEMENT_RESULTS.md).

## Implementation Ledger

| Phase | Implemented capability | Default |
|---|---|---|
| 9 | Create-only public-target-excluding development/final preparation, target/user disjointness checks, fixed scenario/category strata, provenance hashes, and expanded failure taxonomy | Evaluation tooling |
| 10 | Pre-expensive-retrieval sufficiency decision with `retrieve`, bounded sparse `minimal_probe`, and separately gated `clarify_first` | Disabled |
| 11 | One-pass uncertainty choice between D30 and D60, hard D60 ceiling, and per-session escalation budget | Disabled |
| 12 | At most two typed intent hypotheses sharing one total candidate budget; item/product-type/attribute negative-feedback scope | Disabled except feedback diagnostics/safe generic retraction |
| 13 | Structured semantic question goals, duplicate-goal suppression, answer-productivity classification, and positive-value question gate | Disabled |
| 14 | Neural margin/pair diagnostics, grouped Platt calibration, previous-turn calibrated cascade signal, and fixed-budget one-model comparison | Tooling/diagnostics; behavior disabled |

No feature reads targets, sample IDs, scenario labels, or future turns at runtime.
Every failure path retains legal selected-style fallback behavior.

## Frozen Unseen Development Evidence

The preparation command selected targets before any agent inference, excluded all
200 public targets, and created 80 development plus 40 sealed-final sessions. The
development hash is
`3cba45ace6494f28e1fdc707f57a335b9f028314b9a3f773f666cb50992a7fae`.
The final split was unopened during this roadmap experiment. It was subsequently
consumed by the [pipeline refinement evaluation](PIPELINE_REFINEMENT_RESULTS.md);
it is no longer an untouched holdout. See [current status semantics](DATASET_STATUS.md).

One primary ablation at a time was run on the 80-session development split:

| Run | HitRate@10 | MRR | MTTC | TechnicalScore | Delta | p95 | Tokens | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Selected D30 | 0.887500 | 0.636781 | 3.300000 | 0.788784 | — | 0.355s | 874,569 | Control remains selected |
| Sufficiency minimal probe | 0.887500 | 0.619053 | 4.437500 | 0.760716 | -0.028068 | 0.353s | 846,850 | Reject promotion; turn cost erased savings |
| Uncertainty D30/D60 cascade | 0.887500 | 0.637822 | 3.300000 | 0.789097 | +0.000313 | 0.362s | 895,866 | Keep gated; below +0.01 threshold |
| Two-hypothesis retrieval | 0.862500 | 0.628100 | 3.562500 | 0.768430 | -0.020354 | 0.340s | 896,668 | Reject promotion; buying and browsing hits regressed |
| Semantic value-gated dialogue | 0.737500 | 0.510719 | 5.175000 | 0.638466 | -0.150318 | 0.353s | 774,114 | Reject promotion; simulator recovery regressed sharply |

The source remained unchanged during the suite and every arm recorded zero
fallback turns. These are synthetic unseen-target development results, not
organizer-private evidence. Since no arm cleared the registered `+0.01` gate, the
sealed final split was not consumed by this experiment. This is a historical
statement, not the split's current status.

## Calibration Evidence

A separate selected D30 trace reproduced `0.788784` on the same development rows
with 255 turn records and zero fallbacks. Five-fold grouping by synthetic user
produced:

```text
Platt slope                  2.0
Platt intercept             -2.0
0.5-probability margin       1.0
Out-of-fold log loss         0.469055
Out-of-fold Brier score      0.152535
Out-of-fold ECE              0.108166
```

This is calibration infrastructure evidence, not proof that margin-based
escalation improves ranking. The near-neutral uncalibrated cascade remains gated;
the calibrated previous-turn threshold requires a new registered ablation rather
than retroactive tuning on these results.

## Release Compatibility Verification

After all roadmap code was integrated, the unchanged selected entrypoint
reproduced the established 200-session public result exactly:

```text
HitRate@10       0.895000
MRR              0.613746
MTTC             3.245000
TechnicalScore   0.786724
Fallback turns   0
```

The authored private-like pack also retained 19/19 passing assertions with zero
failed or unverified assertions, API errors, and fallback turns. The complete unit
suite passed 444 tests; Ruff and `pip check` passed. These checks establish release
compatibility, not a new private-performance claim.

## Practical Interpretation

- The selected D30 architecture generalized well enough to remain the reliability
  choice on this synthetic unseen-target check.
- Asking before full retrieval can reduce neural work, but in the official
  simulator the extra/cheaper turn materially harms MTTC and ranking evidence.
- More retrieval interpretations are not automatically better: dividing the fixed
  budget weakened strong lexical buying behavior.
- Semantic non-repetition is a correctness improvement, but coupling it to a
  strict positive-value gate suppressed simulator disclosures and is not a release
  policy.
- D60 should be spent only when a better calibrated trigger demonstrates a
  practical unseen gain. D120 remains public demonstration evidence only.
