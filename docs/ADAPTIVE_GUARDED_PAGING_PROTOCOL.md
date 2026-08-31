# Adaptive shortlist versus adaptive shortlist plus guarded paging

Registered before score measurement. Base: `4895be12aa5f559de2b314445a75fec5cdf32c63`.
This is an opt-in experiment; the submission entrypoint is unchanged.

## Frozen arms

- Control: Chef's current default adaptive shortlist, including tentative ambiguity handling.
- Candidate: the identical default plus bounded presentation-only paging.
- Preserve every underlying search result, ranking score, question and output width.
- Observe the existing top-100 ranked question context; no extra retrieval or reranking.
- Advance from the first repeated top-10 membership, using highest-ranked unseen items.
- Clear exposure and return the base shortlist after any active semantic-state change,
  or an explicit correction/override even if it restates an existing value. A changed
  top-10 membership independently returns the base shortlist, retaining exposure history.
- Paging operates within the baseline's safe/known-violation quotas. Never replace a
  compatible displayed product with a known contradiction merely to show something new.
- Keep empty responses empty. On exhaustion, fill from the same safety tier in original
  ranking order. Requests retried with the same turn/message/top_k never advance paging.
- State is session-local, resettable and bounded by the existing ten-turn/top-ten API.

This is a conservative adaptation of earlier guarded paging to variable-width lexical
output, not a claim that toggling the legacy neural agent's config enables Chef paging.

## Measurement

Run fresh matched arms on these already-consumed datasets, with no tuning between runs:
public (200), Cycle 5 screening/confirmation/validation (160/80/80), Cycle 3 screening
(160), and robustness v1 screening (160). No sealed dataset is opened.

Use the unchanged official evaluator and preserve hashes for source, configuration,
catalog, datasets, and evaluator. Recompute aggregates from session outcomes. Report
HitRate, MRR, MTTC, TechnicalScore, per-scenario outcomes, paired target gains/losses,
paired bootstrap score intervals (5,000 resamples, seed 20260831), model tokens,
errors/fallbacks, and descriptive timings. Timings are not a controlled latency study.

Report repeated outputs in scored traces, with denominators. These traces can end at
different times, so additionally replay the same full ten-turn conversations on a
preselected 40-session public subset: 16 buying, 16 browsing, 6 override, 2 boundary;
within each scenario choose by SHA-256(sample_id), before inference. Continue generating
ordinary simulator replies after a target hit. This audit is not an official score.
Both presentations use the exact same underlying result on every replayed turn.

Measure adjacent exact repeats, adjacent set repeats, repeat product exposures within
semantic epochs, unique products per session, longest repeated-slate streak, and
re-showing products immediately after explicit rejection. Split repeats caused by
intent-reset replays from repeats without a reset. Authored cases cover stalled
dialogue, same-value overrides, relaxation, polarity/category changes, small pools,
contradictions, retries, and session isolation.

Known pre-measurement limitation: the existing parser records some rejection
paraphrases (e.g. “Those options aren't right”) as new clarification evidence. The
semantic-change guard therefore replays the base once on that first utterance;
repeating it then pages. The canonical “options are not quite right” is not stored
as preference evidence and can page immediately. This experiment does not repair
the parser or relax its semantic reset to conceal that limitation.

## Interpretation

No automatic promotion or push. Recommend adoption only if all six datasets preserve
both TechnicalScore and target hit count, equal-horizon non-reset repeated slates fall,
and behavior/integrity checks pass. Otherwise report the trade-off or reject this
particular paging design. Do not change thresholds after inspecting score results.
All evidence remains public/local synthetic, not organizer-private or real-user proof.
