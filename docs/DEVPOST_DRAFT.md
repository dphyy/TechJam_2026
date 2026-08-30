# Mercury: search that can change its mind

Draft only; not submitted to Devpost. Confirm the team and replace the video placeholder before submission. No video upload has been performed.

## Inspiration

Shopping conversations change. A person starts with a black leather bag, switches to blue canvas, and still needs an adjustable strap. A useful search system should preserve the strap requirement, retract the old color and material, and avoid treating missing catalog information as proof that a product is unsuitable.

## What it does

Mercury compiles the conversation into a small, reversible preference record and searches the supplied product catalog. It shows ranked product IDs while asking a useful follow-up question. It records where each preference came from, handles corrections and exclusions, and distinguishes supported facts from unknown metadata. A compact local model can rerank the strongest lexical candidates; a sparse fallback works without model files or credentials.

## How it works

The backend uses field-aware lexical retrieval, source-linked conversational state and a conservative contradiction guard. A pinned local MiniLM reranker orders the strongest 30 candidates; simple questions are capped at four, with Top-10 recommendations. We built dense retrieval, catalog-neighbor contrast evidence and rank-aware question policies as independent experiments, then disabled them when their gains did not justify inclusion. See the [report](../REPORT.md) for the frozen choice, exact metrics, negative results and resource disclosure.

The submitted interface is the participant kit's Python `Agent`, not a shopping frontend or a hosted service. It does not modify the official evaluator/catalog, reconstruct private targets or join users to external purchase histories. No model training or live inference API is required.

## What is distinctive

The contribution is an evidence-led combination of reversible query state, unknown-safe constraints and measured question/ranking decisions in a small offline backend. We do not claim that BM25, neural reranking, contrastive retrieval or conversational memory were invented here. The compelling demo is a visible correction that retracts only the affected facts, followed by a judge-facing evidence viewer generated from real agent calls.

## Challenges and lessons

More components did not consistently mean better search. Positive-evidence boosts and dense retrieval hurt the initial strong baseline. Several sophisticated question/slate experiments failed to justify their added complexity. Public simulated conversations also differ from real shoppers: they do not establish real purchase uplift, and the held-out public subset is too small to certify private-test performance.

## Results and next steps

| Evidence | HitRate@10 | TechnicalScore |
|---|---:|---:|
| Current selected paging release, whole public development set | 97.00% | 0.839176 |
| Frozen selected agent, 160 development sessions | 88.75% | 0.775118 |
| Frozen selected agent, 40 reserved public sessions | 92.50% | 0.833146 |
| Selected agent, descriptive whole-public offline reproduction | 89.50% | 0.786724 |
| Original starter, same whole-public set | 12.50% | 0.106710 |

The current 30 August reproduction found 194/200 public targets, with MRR 0.645919, MTTC 2.98, p95 turn latency 0.555 seconds and no fallbacks. The improvement over the historical fixed-slate release comes primarily from paging an unchanged slate late in the conversation. Both frozen 30- and 60-candidate variants previously found 37/40 reserved targets; the 30-candidate version was selected for equal hit rate, slightly higher score and approximately half the reranking work. Runtime paid API/compute cost was US$0, excluding existing hardware and utilities.

These numbers are consumed public simulator evidence, not a private-test forecast or real purchase uplift. Broader correction phrasing and generic negative-feedback handling are now regression-tested. A preregistered low-margin ranking experiment was rejected after MRR and score declined, so the public entrypoint was not changed. Next steps are broader independent conversational testing and final host/data-packaging confirmation.

## Attribution and team

- Participant task, official evaluator and catalog: TikTok TechJam 2026 participant kit, pinned commit in the report.
- Dataset attribution: `DATA_ATTRIBUTION.md`; confirm redistribution terms separately.
- Pretrained models and exact revisions/licenses: `docs/MODELS.md`.
- Project owner: Saai Aravindh Raja.
- Actual team roster and per-person contributions: [confirm before submission; do not invent contributions].

## Publication fields

- Working repository: [TechJam_2026](https://github.com/dphyy/TechJam_2026)
- Public submission source: [owner-approved public URL after final publication; not yet authorized]
- Three-minute YouTube demo: [public video URL after approved upload]
- Build-window contribution record: [significant work completed during the official window]

Preparation began on 26 August 2026. Preserve that fact and confirm eligibility treatment under the event rules before submitting.
