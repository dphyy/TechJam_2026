# Three-minute backend demonstration

Lead with one simple claim: Mercury can change its mind without forgetting what still matters, and it will not invent evidence when catalog metadata is missing. This is a preparation build, not an uploaded submission or a claim of first-place novelty.

## Current recommended walkthrough

Generate the judge view from the real selected agent in a new directory:

```bash
.venv/bin/python -m demo.showcase \
  --results docs/current-results.json \
  --output artifacts/judge-showcase
```

Open `artifacts/judge-showcase/index.html` in a browser. Keep `evidence.json` beside it for technical questions.

| Time | Show | Say |
|---|---|---|
| 0:00–0:25 | Hero, 50,000-product count and current metric cards | “Mercury is an offline search backend for changing shopping intent. These are real agent outputs over the supplied catalog, not a mocked storefront.” |
| 0:25–0:55 | Turn 1: black leather shoulder bag, active preference chips and catalog IDs | “It turns the request into a source-linked preference ledger and returns legal product IDs. Catalog mentions are labeled supported; missing details remain unknown.” |
| 0:55–1:30 | Turn 2: blue canvas correction; retracted black/leather and retained shoulder bag/adjustable strap | “A replacement changes only the affected facts. The old color and material are visibly retracted, while the bag type and strap requirement survive.” |
| 1:30–1:55 | Expand “Why this response?” and one product evidence list | “Judges can inspect the retrieval query, buying/browsing route, feedback scope, page, fallback state and the evidence boundary behind each result.” |
| 1:55–2:15 | Turn 5: reject slate and show paging/result change | “Generic negative feedback does not pollute the query. When the ranking is unchanged late in a session, Mercury advances to a new result page instead of repeating a failed slate.” |
| 2:15–2:40 | Current result and margin-fusion rejection | “On the consumed 200-session public development set, it finds 194 targets: 0.97 HitRate@10 and 0.839176 TechnicalScore. A new low-confidence fusion idea reduced MRR, so it was measured, documented and left disabled.” |
| 2:40–3:00 | Evidence boundary and reliability notes | “The run had zero fallbacks and zero agent errors. Missing model assets still produce a legal sparse fallback. Public data is development evidence, not a prediction of the private set or real purchases.” |

Do not claim that every recommendation satisfies every preference. The page intentionally exposes `unknown` when the catalog cannot establish a fact. Do not call TechnicalScore “accuracy”; HitRate@10 is the 97% accuracy-like metric.

## Historical Cycle 2 recorded evidence

The final cycle-2 replay is `artifacts/cycle2/alternatives-demo-v2/`: `replay.cast`, `transcript.txt`, `responses.json`, `manifest.json` and a labeled invented catalog. It records 24 real API calls under all three controls, with actual local model inference and networking denied. All response contracts pass; all six agents close; no fallback occurs. Source, model, catalog and configuration inventories match before and after.

Generate a new demonstration, not another validation run, with an unused output directory:

```bash
.venv/bin/python -m demo.alternatives --catalog data/catalog.jsonl --selected-mode grouped --output artifacts/local-alternatives-replay
```

`--selected-mode` controls narration only. Frozen, parse-only and grouped all execute the same three preregistered real-catalog exchanges and the invented three-product exchange. Full responses and source evidence are retained. The `.cast` is a narration-paced 180-second terminal recording, not an MP4, a public video or real-time latency footage. Show the separately recorded response times.

The older `artifacts/demo-final/` and `demo.replay` preserve the previous baseline demonstration. They are historical evidence, not the cycle-2 comparison.

## Component-qualified evidence proof

This small, authored capability replay is separate from the consumed Cycle 2 fixtures
and any technical-score run. It uses the exact config-gated role-evidence candidate,
a control that differs only by `role_evidence=false`, and a four-row invented jacket
catalog. It records the ordinary API's diagnostics, including the local catalog span
that earns support. Generate it only in a new output directory:

```bash
.venv-repro/bin/python -m demo.role_evidence --output artifacts/local-role-evidence-replay
```

The replay demands one direct `description` witness for `leather outer shell`; leather
elbow patches and cross-field co-occurrence must not receive one. It then changes the
material to canvas and clears material preference, requiring role evidence to remain
empty on both turns. The enabled configuration is independently replayed twice, and
the runner rejects nondeterministic state/evidence/ranking diagnostics, fallbacks,
bad API responses, source/config/catalog/model drift, or output-directory overwrite.
Its `manifest.json`, `responses.json`, `transcript.txt`, and invented catalog are a
bounded source-span and correction-safety proof only, never score, hidden-set,
shopper, or organizer-private performance evidence.

## Historical Cycle 2 narration and shots

Use the saved responses for API shots and [the final report](CYCLE2_RESULTS.md) for result shots. Do not fabricate a frontend, successful purchase or extra catalog evidence.

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | API contract and 50,000-product catalog | “Mercury turns a changing shopping request into real catalog IDs. Here is one decision it must preserve: either cotton or linen is acceptable.” |
| 0:20–0:50 | Real shirt exchange, both material options in active state | “These are actual returned products. The two materials form one acceptable choice set, not two simultaneous requirements.” |
| 0:50–1:15 | “Actually, no linen”; query becomes shirts cotton; linen is an exclusion | “The correction removes positive linen, retains cotton and the shirt requirement, and rebuilds the ranking. The result remains tied to catalog facts.” Do not claim every returned item is verified linen-free. |
| 1:15–1:55 | Clearly labeled invented catalog; same query/candidates, different guard | “This controlled example exposes the bug. A cotton shirt explicitly says linen-free. Independent constraints penalize it for lacking linen. Grouped alternatives remove that false penalty: rank two becomes rank one. Missing evidence stays unknown.” |
| 1:55–2:25 | Three-control target and capability tables | “All controls found 179 of 200 public targets, score 0.786724. On a different locked 32-session set, all found 31, score 0.838032. This repair did not raise those scores. All controls also passed 46 of 51 locked capability checks.” |
| 2:25–2:45 | Source freeze, consumption receipts, tests and resource row | “The source was frozen before validation; each locked comparison ran once. We have 347 passing tests. Grouped validation p95 was 0.420 seconds, with a separately measured 19.188-second cold start and no paid inference.” |
| 2:45–3:00 | Offline fallback receipt and remaining-limitations panel | “Missing or invalid model files fall back safely with networking disabled. The controlled repair works; real-catalog benefit and broader body-versus-component understanding remain open.” |

## Claims and judge questions

**What is actually new here?** A bounded integration of explicit alternatives, source-linked reversible state and conservative catalog evidence. Boolean logic, truth maintenance and reranking have prior art. The claim is the verified behavior of this implementation, not a world-first algorithm.

**Did it improve the score?** No incremental target-score gain in this cycle. The earlier starter-to-backend public comparison is 0.106710 to 0.786724, as documented in the original report. The 0.838032 number comes from a different 32-session sample and was already achieved by frozen Mercury.

**Is the striking example a real product?** No. It is an explicitly invented three-row catalog, executed through the real agent. The three fixed real-catalog probes did not yield a source-verifiable grouped-guard benefit; all are retained. Do not crop away that disclosure or imply that the example establishes prevalence.

**Does unknown evidence count as a match?** No. Unknown prevents an unsupported contradiction penalty; it does not certify suitability. “Canvas only” also does not prove absence of leather trim. Show the remaining real mixed-material result and require an explicit exclusion plus sufficient evidence before making a purity claim.

**What still fails?** Two locked body/component comparisons rank the wrong item first, and three ordinary-query comparisons lack a retained comparator. All three controls share those failures. Those checks were not fixed or rerun after validation.

**Why keep the grouped repair?** It passes the registered truth-table/correction tests, fixes a controlled false contradiction relative to parser-only, and passes the no-regression/resource gates. Those are narrow engineering grounds, not evidence of broad quality superiority.

**What runs without a cloud service?** The pinned local reranker and sparse fallback. Network-denied inference and healthy/missing/invalid asset paths were tested after installation. A fresh air-gapped dependency installation and organizer hardware are not yet verified.

## Before recording or publication

- Read the actual returned titles and keep claims aligned with source evidence. Do not disguise weak matches or the invented example.
- Use the final-source measurements, with sample sizes, controls and uncertainty. Tiny invented-catalog latency is not full-catalog performance.
- Keep preparation dates visible and record significant official-window work separately.
- Keep credentials, personal profiles and unrelated desktop content out of the recording.
- Video export and an approved public YouTube upload are still pending. No encoder was on the checked shell PATH; the existing terminal replay is not an uploaded video.
- Keep publication, video upload and submission as explicit owner-approved steps. See [the release checklist](RELEASE_CHECKLIST.md).
