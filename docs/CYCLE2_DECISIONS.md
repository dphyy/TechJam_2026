# Cycle 2 decision ledger

## Source-qualified roles: reject as the lead experiment

Before implementing the role mechanism, the frozen runtime was characterized on the two pre-registered role/scope development cases. The actual local MiniLM model and selected blend weight were used, with the unchanged state, retrieval and constraint ranking. Both cases passed: `d01` ranked the complete bag before the replacement strap; `d02` ranked the wool-body jacket before the nylon-body jacket. There were no model fallbacks; measured input tokens were 55 and 70 respectively. The sparse control also passed both cases.

The registered gate requires at least 20 percentage points improvement over frozen Mercury on this subset. With the reference already at 2/2, the maximum improvement is zero. The lead experiment is rejected without building a larger role parser or weakening the gate. This result does not prove that role errors do not exist; it shows that the selected test cannot earn the proposed superiority claim. No validation data was opened.

## Bounded alternatives: selected fallback hypothesis

The preselected fallback in the research funnel was explicit Boolean alternatives. A separate development characterization found a concrete failure: “Either a brown or black belt works.” leaves no active preferences because the broad no-preference pattern consumes the whole clause. Existing ordinary “blue or green” support does not cover this case. The mandatory contradiction guard also treats every hard alternative independently; a product explicitly satisfying one option can be penalized for not satisfying the other.

The next probe is deliberately narrower: preserve explicit same-attribute alternatives and evaluate their contradictions as a group. Compare the frozen runtime, a minimal parser-only repair, and parser plus grouped constraints. This is conventional Boolean semantics, not a new research algorithm. Its claim must be limited to measured correctness and correction handling.

The original experiment registration remains unchanged. The prospective fallback protocol is in `docs/CYCLE2_ALTERNATIVES_PROTOCOL.md`; all outcomes obtained before that registration are the characterizations stated above, not untouched evidence.

## Matched frozen-code reruns

The original source was exported from `99095211a1732eb750610bd59610f215dee3136f` into an isolated archive. All 52 source hashes in the original finalist manifest match that archive and both new run snapshots. Source hashes remained unchanged during each run. The outer repository commit recorded by the runner is not the archive's algorithm version; the archived source hashes establish that provenance.

| Development set | Hits | TechnicalScore | Warm p95 | Peak RSS |
|---|---:|---:|---:|---:|
| Original public, 200 sessions | 179/200 | 0.786724 | 0.499 s | 986 MiB |
| New target families, 32 sessions | 27/32 | 0.751053 | 0.391 s | 817 MiB |

Both runs use the actual existing MiniLM model, report no model fallbacks or agent-error turns, and incur no paid API cost. They are different development sets, not an improvement comparison. The original public score is reproduced exactly. Neither run opens validation.

Local evidence: `runs/cycle2-frozen-public-development` and `runs/cycle2-frozen-new-development`. Their result SHA-256 digests are respectively `eeba444682d430d42bd34e87f58668187f82c21486b4dd791890c6176d85ec5e` and `bc53e7bbdaeb21b976f6c673b6cc30d47577f75d3ba75f567ebc74c36442d45d`. Target-level traces remain local.
