# Cycle 4 screening results

## Structured reranker query

Recorded 29 August 2026 against a source-matched selected control on the already
consumed 160-session Cycle 3 screening pack. This is development evidence only; no
Cycle 3 confirmation or validation target was opened.

| Arm | TechnicalScore | Hit@10 | MRR | MTTC | Warm p95 |
|---|---:|---:|---:|---:|---:|
| Source-matched flat control | 0.811786 | 0.925000 | 0.635119 | 3.062500 | 0.322528 s |
| Structured state query | 0.772979 | 0.893750 | 0.573681 | 3.300000 | 0.332805 s |

Both runs had valid IDs, no source drift, zero fallbacks, and zero agent-error turns.
The paired TechnicalScore delta was `-0.038807` with a 10,000-resample bootstrap
interval of `[-0.063548, -0.016214]` (seed `20260826`); HitRate@10 changed by
`-0.031250`. The predefined decision tool marked both required objectives as
regressions.

The labelled state query therefore worsened ranking rather than clarifying it. It is
rejected, its runtime implementation was reverted, and its live configuration was
removed to prevent an unsupported release setting. The result does not support a
claim about organizer-private performance.

## Stability-protected 60-candidate reranking

Recorded 29 August 2026 against a fresh source-matched selected control on the same
already consumed 160-session Cycle 3 screening pack. This is development evidence
only; no Cycle 3 confirmation or validation target was opened.

| Arm | TechnicalScore | Hit@10 | MRR | MTTC | Warm p95 |
|---|---:|---:|---:|---:|---:|
| Source-matched selected control | 0.811786 | 0.925000 | 0.635119 | 3.062500 | 0.321107 s |
| Stable 60-candidate reranker | 0.812522 | 0.925000 | 0.637158 | 3.056250 | 0.610834 s |

Both runs had valid IDs, no source drift, zero fallbacks, and zero agent-error turns.
The candidate improved the screening score by `+0.000736` while retaining HitRate@10,
but it did not reach the registered `+0.010000` minimum. Its p95 was `1.90x` the
matched control and remained inside the `2x` guard; the improvement is still too small
to distinguish from an ordinary screening fluctuation. The decision tool classified
both required comparisons as inconclusive.

The stable-tail implementation and configuration were therefore reverted rather than
combined with another arm. The result does not support a claim about organizer-private
performance.

## Concessive-negation parser repair

Recorded 29 August 2026 against the source-matched selected control on the same
consumed 160-session Cycle 3 screening pack. A shopper phrase beginning “no matter
if/whether” is concessive, not an exclusion: treating its opening `no` as a negation
created unsupported hard constraints from descriptive source language. The repair
scopes that phrase before ordinary negation parsing and retains direct `no` and
`without` exclusions.

| Arm | TechnicalScore | Hit@10 | MRR | MTTC | Warm p95 |
|---|---:|---:|---:|---:|---:|
| Source-matched selected control | 0.811786 | 0.925000 | 0.635119 | 3.062500 | 0.359716 s |
| Concessive-negation repair | 0.817786 | 0.931250 | 0.641369 | 3.012500 | 0.326734 s |

The run had valid IDs, no source drift, zero fallbacks, and zero agent-error turns.
It recovered one previously hard-excluded source-supported item by admitting it to the
existing 30-pair reranker prefix; the unchanged reranker then ranked it first. The
single-run `+0.006000` TechnicalScore delta is below the registered `0.010000`
stability threshold, so it is not evidence for a broad score-lift claim. The correction
is retained as a general language-correctness repair, not combined with rejected
ranking arms, and any later score candidate must use this source as its control.

## Component-qualified source evidence preflight

The fixed role-evidence candidate passed fresh authored mini-catalog cases: direct
whole-product phrases such as `leather outer shell` receive a source-field witness,
while a leather handle, lining, patch, cross-field co-occurrence, missing evidence,
and inactive or negative requests remain unknown. The candidate also requires a
currently active matching material preference, so changing leather to canvas or
neutralizing material retracts the witness even if the broad `leather outer shell`
phrase remains in state. It makes no hard exclusion and records its supporting source
span in diagnostics.

The consumed 160-session screening traces contain zero active material-plus-whole-role
requests within the candidate's bounded vocabulary. It therefore cannot change that
screening result, so no full scorer run was spent and it is not selected for the
technical-score configuration. The separate
`configs/cycle4_role_evidence.json` capability configuration remains reproducible for
demonstration and future independently authored evaluation; it does not justify a
claim about organizer-private performance.

## Composition-qualified evidence screening

Recorded 29 August 2026 against a fresh source-matched selected control on the
already-consumed 160-session Cycle 3 screening pack. The one registered candidate
applied a single `+0.010` post-reranker adjustment only to a current dependent
`N% material` request with a matching active material and a direct same-field source
span. It did not change retrieval, the 120-candidate ceiling, the 30-pair reranker,
model inputs, question policy, output size, or hard constraints.

| Metric | Matched control | Composition candidate | Delta |
|---|---:|---:|---:|
| TechnicalScore | 0.817786 | 0.819057 | +0.001271 |
| HitRate@10 | 0.931250 | 0.931250 | +0.000000 |
| MRR | 0.641369 | 0.645605 | +0.004236 |
| MTTC | 3.012500 | 3.012500 | +0.000000 |
| Efficiency | 0.798750 | 0.798750 | +0.000000 |
| Warm p95 seconds | 0.324304 | 0.342795 | +0.018491 |
| Reranker prompt tokens | 1,724,323 | 1,724,323 | +0 |

Both manifests report unchanged source during the run, zero fallback turns, and zero
agent-error turns. The candidate emitted direct composition witnesses on 142 of the
471 recorded turns and changed the final diagnostic rank order on 130 turns, so the
mechanism was active and discriminative rather than inert. Scenario HitRate@10 did
not decline; the MRR gain occurred in browsing and intent-override sessions.

The `+0.001271` TechnicalScore increase is below the predeclared `+0.010000`
promotion threshold. The candidate is therefore rejected from confirmation,
validation, and `configs/selected.json`; it must not be tuned, combined with another
arm, or presented as a broad/private-test improvement. The selected score remains
0.817786 on this source-matched screening pack.

## Source-alias sparse parity

Recorded 29 August 2026 against a fresh source-matched selected control on the
already-consumed 160-session Cycle 3 screening pack. The candidate preserved the
canonical state query and added a fixed, separate 15% sparse route only for exact
non-canonical parser aliases from current positive source text, such as a shopper's
`trainers` wording for the canonical `sneakers` fact. It added no external synonym or
inferred relation and retained the 120-candidate ceiling and 30-pair reranker.

| Metric | Matched control | Source-alias candidate | Delta |
|---|---:|---:|---:|
| TechnicalScore | 0.817786 | 0.817786 | +0.000000 |
| HitRate@10 | 0.931250 | 0.931250 | +0.000000 |
| MRR | 0.641369 | 0.641369 | +0.000000 |
| MTTC | 3.012500 | 3.012500 | +0.000000 |
| Efficiency | 0.798750 | 0.798750 | +0.000000 |
| Warm p95 seconds | 0.318426 | 0.318308 | -0.000119 |
| Reranker prompt tokens | 1,724,007 | 1,718,761 | -5,246 |

The target-blind preflight observed active aliases on 123 of 471 shopper turns, a
nonempty alias route on all 123, and a changed bounded retrieval prefix on 110 turns.
The actual paired replay changed retrieved order on 123 turns, the reranker prefix on
86, and final rank order on 110; it was therefore active and not an inert route. Both
manifests report unchanged source during the run, zero fallback turns, and zero
agent-error turns.

Despite that activity, every registered score metric was identical. The candidate did
not clear the predeclared `+0.010000` promotion gate and is rejected from confirmation,
validation, and `configs/selected.json`. Its fixed weight must not be retuned or
combined with a rejected arm; the selected score remains 0.817786.

## Metadata-label parser audit

One final target-blind failure-mode audit found generic pasted catalog labels such as
`Package Dimensions`, `Department`, and `Is Discontinued By Manufacturer` in 279 of
the 471 recorded screening turns. `package dimensions` persisted as an open-vocabulary
state value on 22 turns. This is not promoted to a score arm and no scorer run was
spent: removing labels would alter query/state serialization and is coupled to the
evaluator's generated catalog-field templates. The audit found no remaining
independent, generalizable candidate outside the already rejected retrieval,
admission, reranker, question/slate-policy, broad-evidence, role, and composition
families.

## Exact source-phrase route preflight

Recorded 29 August 2026 on the already consumed Cycle 3 screening traces, before a
full evaluator run. The route formed bounded, sanitized FTS phrases from active source
clauses and fused matches with the unchanged broad/scoped candidate pool. Its declared
mechanism was recovering the two target products absent from the 120-item selected
candidate pool.

It recovered neither product into that pool. The catalog contains many product rows
with the same template feature text, so the phrase route's low-weight reciprocal-rank
contribution did not overcome the ordinary broad/scoped candidates. The route therefore
failed its causal candidate-recall check and was discarded before score measurement;
its implementation and configuration were never selected. No confirmation or
validation target was opened.

The result does not support a claim about organizer-private performance. The remaining
Cycle 4 work is the independently gated component-qualified evidence capability.
