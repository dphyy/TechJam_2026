# Three-minute backend demonstration

Lead with one shopping decision and its evidence: “cotton or linen” must not become “cotton and linen”, and “actually, no linen” must retract the right option. Show actual API results, the controlled repair and the unchanged benchmark. This is a preparation build, not an uploaded submission or a claim of first-place novelty.

## Recorded evidence and reproduction

The final cycle-2 replay is `artifacts/cycle2/alternatives-demo-v2/`: `replay.cast`, `transcript.txt`, `responses.json`, `manifest.json` and a labeled invented catalog. It records 24 real API calls under all three controls, with actual local model inference and networking denied. All response contracts pass; all six agents close; no fallback occurs. Source, model, catalog and configuration inventories match before and after.

Generate a new demonstration, not another validation run, with an unused output directory:

```bash
.venv/bin/python -m demo.alternatives --catalog data/catalog.jsonl --selected-mode grouped --output artifacts/local-alternatives-replay
```

`--selected-mode` controls narration only. Frozen, parse-only and grouped all execute the same three preregistered real-catalog exchanges and the invented three-product exchange. Full responses and source evidence are retained. The `.cast` is a narration-paced 180-second terminal recording, not an MP4, a public video or real-time latency footage. Show the separately recorded response times.

The older `artifacts/demo-final/` and `demo.replay` preserve the previous baseline demonstration. They are historical evidence, not the cycle-2 comparison.

## Narration and shots

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
- Keep the source private and the README empty until the owner authorizes the remaining publication steps. See [the release checklist](RELEASE_CHECKLIST.md).
