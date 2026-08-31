> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Explicit alternatives: controlled engineering result

Preparation work dated 26 August 2026. This is a backend experiment, not a competition submission, a private-leaderboard score or a first-place prediction.

## Release decision

Select the already frozen **grouped** configuration as a narrow correctness repair. It passed every registered developer invariant, ordinary-target regression floor, locked no-new-failure gate and local resource limit. Selection copies the measured grouped configuration exactly; no source or parameter was retuned after validation.

There is **no measured target-score improvement and no incremental locked capability pass**. All three controls score 0.786724 on the released 200 sessions, 0.751053 on the new 32-session development set and 0.838032 on the different 32-session validation set. Validation is 46/51 capability assertions for every control. The higher validation score is a different sample, not an improvement over the baseline.

The developer truth table and labeled invented-catalog replay demonstrate the false-contradiction repair. The three fixed real-catalog probes did not exhibit a verified grouped-guard benefit. This supports shipping the bounded repair, not claiming general superiority or a first-place innovation. See the [selection receipt](cycle2-selection.json) and [aggregate evidence](cycle2-summary.json).

The contribution is bounded correctness: preserve a shopper's explicit acceptable alternatives, let later corrections retract them coherently, and avoid penalizing a product unless every option in a required choice set has observed contradictory evidence. This is conventional Boolean and state-management engineering, not a new ranking algorithm.

## What was built

The existing source-linked preference ledger now represents a positive, same-attribute choice set. “Must be cotton or linen” is one requirement with two acceptable options. A separate “no leather” remains an independent exclusion. The common constraint guard runs before candidate truncation and after the existing neural reranker.

| Evidence for the live options | Group interpretation | Contradiction penalty |
|---|---|---|
| At least one supported option | Supported | None for this group |
| No support, at least one unknown | Unknown | None for this group |
| Every option explicitly contradicted | Contradicted | Apply the existing guard |

Unknown is not a claim that the product fits. Absence of a catalog fact cannot prove that it violates a requirement. The guard changes ranking; it does not delete catalog rows. Negative requirements remain independently binding.

Corrections work against the active ledger, in clause order. An ordinary restatement preserves group identity and its requirement strength. An explicit selection, including “only cotton”, retires the other options. Rejecting one option keeps the surviving requirement. Clearing color preserves unrelated material/category requirements. Adding an unrelated feature does not erase an existing feature choice. Group changes invalidate ranking caches even when query words remain unchanged.

The parser supports direct links between known facet values. Nested expressions, arbitrary mixed conjunctions, cross-attribute Boolean expressions, component/body scope and unrestricted English understanding are not claimed. Unsupported overlapping lists retain the existing choice set and report a diagnostic while independently valid exclusions still apply. Pre-existing ungrouped behavior remains unchanged; this cycle does not claim a general solution to every use of “only”.

## Controls and unchanged budgets

| Control | Parsing | Constraint interpretation |
|---|---|---|
| Frozen | Original behavior | Independent hard preferences |
| Parse-only | Preserve explicit lists swallowed by no-preference wording | Independent hard preferences |
| Grouped | Same list repair plus explicit choice lifecycle | Positive OR groups; independent negatives and conjunctions |

All three use the same catalog, model revision, four CPU threads, 120 retained candidates, at most 30 cross-encoder pairs, 256 tokens per pair, blend weight 0.75, Top 10 and existing question policy. Dense retrieval, contrast ranking and optional positive evidence boosts remain disabled. No new model, training, paid inference or target lookup is introduced. Runtime receives ordinary messages and an empty profile, never target IDs or scenario labels.

The actual reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Required local model files, including their manifest, occupy 91,820,224 bytes. Downloads and dependency installation are preparation steps; measured inference does not need networking or provider credentials.

## Target recovery

Frozen, parse-only and grouped have identical complete official results within each set:

| Set | Sessions | Hits | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|---:|---:|
| Released public, development | 200 | 179 | 0.895000 | 0.613746 | 3.24500 | 0.775500 | 0.786724 |
| New target families, development | 32 | 27 | 0.843750 | 0.601426 | 3.56250 | 0.743750 | 0.751053 |
| New target families, locked validation | 32 | 31 | 0.968750 | 0.639273 | 2.90625 | 0.809375 | 0.838032 |

Input tokens per control are respectively 2,375,727, 373,228 and 388,332; completion tokens are zero. These are actual local cross-encoder tokens, not a hosted-service bill. All nine final target runs use the same measured source and actual pinned model, with zero startup fallbacks, fallback turns or agent-error turns.

TechnicalScore is the unchanged organizer simulator's `0.5 * HitRate@10 + 0.3 * MRR + 0.2 * clip((11 - MTTC) / 10, 0, 1)`. Misses contribute MTTC 11. A session's first eligible Top-10 hit determines its rank and turn contribution; recommendations before an intent-change gate do not count. The score is not accuracy. See [the scoring guide](SCORING_AND_JUDGING.md).

All 200 released sessions are development data. The earlier 40-session reserve is consumed. The new 32/32 target split uses one target per checked title family, excludes the old public targets and has no cross-split ID/title-family overlap under its stated heuristic. It is generated with the unchanged simulator, not a manufacturer-family certificate, human study or organizer-private set.

Every paired per-session score difference is zero. Against frozen, both candidates have score delta 0, hit-rate delta 0 and paired 95% bootstrap interval [0, 0] on each set, using 10,000 resamples and seed 20260826. An observed all-zero difference is not proof of population equivalence, superiority or organizer-private performance.

There are **zero active OR turns in every target-recovery run**. These sets establish ordinary-behavior regression evidence, not coverage of the new choice semantics. The original `99095211` runtime and corrected-source OFF control additionally match on all 737 development turns: complete responses and official results, queries, preferences, revisions, routes, cache use, rankings, question decisions and tokens. Only timing and additive diagnostic fields are excluded from that parity check.

The public set has 21 misses attributed to ranking/question policy and none to never retrieving the target. New development has one never-retrieved miss and four ranking/policy misses. Validation has one ranking/policy miss. These diagnostics are policy-dependent because questions change subsequent disclosures; they are not independent fixed-query recall experiments.

## Authored capability checks

| Split | Control | Cases passing all assertions | Passed / failed / unverified assertions |
|---|---|---:|---:|
| Development: 8 cases, 10 turns | Frozen | 6/8 | 12 / 3 / 0 |
| Development: 8 cases, 10 turns | Parse-only | 7/8 | 14 / 1 / 0 |
| Development: 8 cases, 10 turns | Grouped | 7/8 | 14 / 1 / 0 |
| Locked validation: 24 cases, 30 turns | Frozen | 19/24 | 46 / 5 / 0 |
| Locked validation: 24 cases, 30 turns | Parse-only | 19/24 | 46 / 5 / 0 |
| Locked validation: 24 cases, 30 turns | Grouped | 19/24 | 46 / 5 / 0 |

Both repairs preserve the brown/black alternatives in development case `d07`, fixing two assertions; neither loses a passed assertion. The ordinary-query comparator remains absent in `d08`, so that assertion remains failed. Development grouped execution contains one active OR turn and no hard-OR turn. Input tokens are 415 for frozen and 411 for each repair. Validation uses 1,658 input tokens per control. Every capability run has zero API errors, fallbacks, health-unverified turns or close errors.

These invented mini-catalog cases were authored before implementation in a separate research scope. They test semantic invariants, not 50,000-product retrieval quality or shopper conversion. Missing ranking comparators are failures; absent diagnostics are unverified, never silent passes. Reported cases require all their assertions, successful responses and verified healthy execution.

Locked results are identical by group for all three controls:

| Capability group | Complete cases | Passed / total assertions |
|---|---:|---:|
| Alternatives and conjunction | 3/3 | 10/10 |
| Body versus component | 1/3 | 1/3 |
| Correction | 3/3 | 11/11 |
| Explicit negation | 3/3 | 4/4 |
| No preference | 3/3 | 9/9 |
| Object versus accessory | 3/3 | 4/4 |
| Ordinary queries | 0/3 | 0/3 |
| Unknown metadata | 3/3 | 7/7 |

Grouped validation has two active OR turns, including **one hard-OR turn**. The hard color-alternatives case lacks explicit contrary-color evidence, so it does not distinguish the grouped contradiction guard from the controls. The soft material-alternatives and ordinary conjunction cases also pass under every control. No new failure appears, but no incremental grouped benefit is independently demonstrated.

The five failures remain visible. In `v05`, both products are retained, but a cotton-body bag with leather handles ranks ahead of the requested leather-body bag with a cotton lining. In `v06`, a silicone ring with a steel accent ranks ahead of the requested steel band. The three ordinary-query cases (`v22`–`v24`) lack at least one comparator in the retrieved/retained list; they fail rather than silently passing or becoming unverified. These are pre-existing limitations in every control. Outcomes were inspected only after all six validation runs; no implementation was changed in response.

## Correctness and operational evidence

All 347 unit/integration tests pass in the fresh `.venv-repro` environment. Ruff, dependency consistency and whitespace checks pass. The grouped guard has a complete nine-cell supported/unknown/contradicted truth table, plus hard/soft groups, independent negatives, correction/retraction, unchanged-query cache invalidation and idempotence tests. These are developer-authored regressions, not held-out accuracy measurements.

Seven local review lenses and an independent findings validator identified six distinct defects. They were fixed before final development comparisons and validation. A separate read-only check confirmed the fixes. Test-first failures and their scope are recorded in [the quality log](QUALITY_LOG.md). No external-provider source-analysis assurance is claimed.

Final full-catalog measurements below are seconds and peak resident MiB (bytes / 1,048,576). Each row is one serial run; timing differences do not establish a causal speedup.

| Set | Control | Cold construction | Warm p50 | Warm p95 | Warm max | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| Public development | Frozen | 17.533 | 0.273 | 0.371 | 0.637 | 990.6 |
| Public development | Parse-only | 16.998 | 0.273 | 0.406 | 0.695 | 1045.1 |
| Public development | Grouped | 17.779 | 0.273 | 0.399 | 0.745 | 979.6 |
| New development | Frozen | 17.819 | 0.256 | 0.356 | 0.437 | 1075.9 |
| New development | Parse-only | 17.950 | 0.269 | 0.400 | 0.710 | 837.7 |
| New development | Grouped | 16.768 | 0.249 | 0.330 | 0.370 | 1109.9 |
| Locked validation | Frozen | 17.535 | 0.279 | 0.470 | 0.696 | 1061.9 |
| Locked validation | Parse-only | 17.626 | 0.303 | 0.533 | 1.189 | 1027.6 |
| Locked validation | Grouped | 19.188 | 0.284 | 0.420 | 0.680 | 1011.5 |

Both candidates pass warm p95 < 1 second and at most twice the matched frozen control, peak RSS < 2 GiB and zero paid inference. The asset check is 152,366,551 bytes for model files plus catalog, below the registered 0.5 GiB asset limit; this is not the total disk footprint of Python, dependencies, caches and evidence. Tiny capability catalogs have separate resource measurements in the aggregate and must not stand in for 50,000-product latency.

Not every turn was subsecond: final parse-only validation reaches 1.189 seconds. An earlier development-v1 measurement reached 1.423 seconds and remains recorded; v2 repeats every control after disclosed review fixes rather than selecting the fastest run. Cold construction is separate from warm response latency.

With OS networking denied and five named provider credentials removed, the final source also completed a full-catalog two-turn correction under healthy, missing-model and invalid-weights conditions. All six responses have ten unique valid catalog IDs and preserve canvas while retracting positive leather and recording its exclusion. Healthy execution loads the actual model (10,359 input tokens); missing/invalid assets produce explicit sparse fallbacks and zero model tokens. Their cold starts are 17.628, 15.505 and 14.153 seconds. Peak process RSS is 805,289,984 bytes. A separate loopback-bind probe confirms network denial; original model assets remain unchanged.

An indefinitely stalled native model kernel still requires an external process timeout. A recorded sparse fallback is degraded execution, not reproduction of the neural score. OS network denial was tested after dependencies were installed; a fresh air-gapped wheelhouse installation and final organizer hardware have not been verified.

## Backend demonstration

The final replay at `artifacts/cycle2/alternatives-demo-v2/` contains 24 actual API calls: three preregistered real-catalog probes, two turns each, under all three controls, plus the same two-turn invented three-product catalog under each control. All response contracts pass, all six agents close, no model fallback occurs, and source/config/catalog/model inventories remain unchanged. Every response and output digest is retained alongside a readable transcript and 180-second narration-paced `replay.cast`.

Real bag, shirt and jacket exchanges demonstrate alternatives, subsequent selection/rejection and retained unrelated context. **No verified real-catalog grouped-guard intervention was observed.** The real selected-control responses took approximately 0.164–0.356 seconds; this small authored demonstration is not the resource benchmark.

The explicitly invented cotton shirt states “Cotton fabric. Linen-free.” with cotton support +0.95 and linen contradiction -0.40. Under the same visible query, active preferences and retrieved IDs, parse-only assigns a 1.019777 contradiction penalty and ranks it second; grouped assigns zero and ranks it first. Separate invented rows show unknown evidence (both signals zero) and contradiction of both options (both -0.40). That is the concrete correctness witness, not a claim about a real shopper or catalog prevalence. The [demo guide](DEMO_SCRIPT.md) supplies a three-minute claim-to-evidence narrative and judge questions. Video encoding/upload remain pending.

The replay retains every fixed probe and all controls. It requires identical visible queries, retrieval IDs and active preferences before claiming a grouped-guard intervention over parse-only. Candidate presence, explicit finite penalties and actual source evidence are required. An invented example is labeled as such; it is not passed off as a real catalog discovery or independent shopper test.

Selecting canvas alone does not prove that an item has no leather trim. Material purity requires an explicit exclusion and adequate catalog evidence. Returned titles and recorded state must be shown honestly, including remaining weak matches.

## Provenance and reproduction

The original frozen runtime is `99095211a1732eb750610bd59610f215dee3136f`; the corrected cycle source is `8964157d701c5cd9d3c642246d38f38db91248a0`. Final development runs use a `git archive` of that source. Runner manifests can record a later enclosing repository commit while documentation is written; exact archived source hashes, not that enclosing commit alone, identify the measured implementation.

The final 60-file source inventory has SHA-256 `6773e77ebb3b04dd9a784ba52ab9a382e28f9f3ac4e317801388828ce2785c63`. The inventory digest hashes sorted compact JSON of source-path/file-hash pairs, not a concatenation of file contents. Source remained identical across all 15 final comparison runs, the offline fault probes and the replay.

The freeze was created at `2026-08-26T13:14:17.214377+00:00`, before validation inference. Its manifest SHA-256 is `efbeb75ad1ea95c4c6ad834a4b9fbd2524cf9573d2f9def713eaffdf77e0d9aa`. Each of the six mode/kind jobs has one consumption marker written before inference and one completed receipt, totaling 12 files. No validation job was retried. All completed by `2026-08-26T13:16:55.985962+00:00`.

Immediately before promotion, the original freeze verifier passed. The [selection receipt](cycle2-selection.json) then records that only the frozen `selected_config` input changed, to the exact pre-frozen grouped bytes (SHA-256 `9fda091879cc4f5e4b8709dce1f9fea8eb3f540d939481ca2e96edf58a30baf3`). All runtime/experiment/test source, other inputs, model files, registered control configs and consumption receipts still match the freeze.

**The original freeze verifier intentionally rejects the promoted checkout**, because it requires the original OFF baseline in `configs/selected.json`. It was not weakened to accommodate the selection. Use the preserved pre-promotion receipt and source snapshot to audit completed validation, and the selection receipt to audit promotion; do not reset the ledger or treat a new output name as another validation budget.

The final development and validation aggregate digests are respectively `5ec6ad58322ece129efbc25c07045a20c282d8786a3ef0c735eae65bfb6c4bd9` and `4c922a773d0a925fa1f7a8af1a10dd28a788520d5314e1104c5debff0ec34ebc`. [cycle2-summary.json](cycle2-summary.json) includes each run's raw-artifact digests, configurations, paired comparisons, group failures and resource measurements. Earlier development versions are retained and labeled; they are not substituted for final-source comparisons.

The catalog and evaluator remain unchanged. Catalog SHA-256: `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`. Original public-set SHA-256: `857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`. Protocols, both split locks, model inventories, source snapshots, complete responses and consumption receipts remain available locally. Source control includes aggregate evidence, not catalog data, model weights or raw target traces.

Use [SETUP.md](SETUP.md) for dependencies, model acquisition and the unchanged official CLI. Named development reproductions must use unused output directories. The one-shot validation guard rejects repeated mode/dataset runs even if an output name changes. Do not relabel consumed validation as an untouched test for further tuning. The local authored packs and provenance artifacts are needed to reproduce their exact comparisons; the aggregate alone is not a substitute for those inputs.

## What the research did and did not establish

The lead role-qualified mechanism was rejected before implementation: frozen Mercury already passed both registered role/scope development cases, making the required 20-point gain impossible on that subset. The gate was not weakened. Explicit alternatives were the single preselected fallback. H2 protected evidence packing and H3 process isolation were not implemented or credited with hypothetical gains.

Source-linked memory, retrieval/reranking, negation handling, truth maintenance and constraint reasoning have substantial prior art. [Amazon's product-search negation work](https://www.amazon.science/publications/improving-the-relevance-of-product-search-for-queries-with-negations), [Doyle's truth-maintenance system](https://www.sciencedirect.com/science/article/pii/0004370279900080) and [QuickXplain](https://aaai.org/Papers/AAAI/2004/AAAI04-027.pdf) preclude treating a new project name as algorithmic novelty. Our evidence concerns this implementation's bounded behavior under fixed budgets.

Prior winners suggest making the consequential decision, working mechanism and proof legible. They do not provide a causal formula for winning. The reviewed record includes the organizer-listed 2023 and 2025 Singapore awards, 2024 regional judged projects and popular-choice awards; some 2024 regional assignments remain inferred. Confidential judging scores and complete independent code/video audits were not available. [2023 results](https://developers.tiktok.com/blog/2023-tiktok-hackathon-challenge-highlights), [2024 SEA results](https://newsroom.tiktok.com/tiktok-supports-singapore-and-sea-talent-in-its-first-multi-market-hackathon?lang=en-SG), [2025 Singapore results](https://newsroom.tiktok.com/tiktok-techjam-2025-concludes-with-winning-innovations-in-singapore?lang=en-SG).

## Submission-readiness verdict

This cycle is complete as **verified preparation engineering**, with a selected correctness repair, reproducible comparisons, reviewed code and a recording-ready backend demonstration. It does not establish first-place novelty, higher target recovery than the strong baseline, broad semantic correctness or real-shopper benefit.

The strongest defensible presentation is a changing shopping requirement, inspectable catalog evidence, a controlled false-contradiction repair, and explicit limits. The largest evidence gaps are a real-catalog grouped benefit, broader body/component understanding, human task success and final-host verification. Future improvements need a new registered experiment and new evaluation data; the consumed packs cannot remain an untouched holdout. The [human-test protocol](HUMAN_TEST_PROTOCOL.md) is proposed, not conducted, and no tester recruitment or organizer outreach has occurred.

The repository remains private, `main` retains the individual commits, and the landing README remains intentionally empty. Public source publication, a public three-minute video, eligibility/team details, significant official-window contributions and Devpost submission remain separate owner-controlled actions. Organizer hardware/model/data policy and the early-brief versus general-rules rubric discrepancy still need confirmation. See [the release checklist](RELEASE_CHECKLIST.md).
