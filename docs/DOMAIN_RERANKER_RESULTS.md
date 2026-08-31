> **Historical comparison — not current release guidance.** Retained to show measured improvements, regressions, rejected approaches or the failure that motivated a change. All logic, scores, test counts and selection statements below apply only to that experiment's recorded source/configuration. For the newest result use [current results](current-results.json); for current runtime/setup use [README](../README.md). Dataset exposure is governed by [recorded consumption](DATASET_STATUS.md), not old “sealed” wording.

# Product-domain reranker results

Phase 11 trained one local `ms-marco-MiniLM-L6-v2` candidate with seed
`20260830`, one epoch, learning rate `1e-5`, maximum length 256, and four
catalog-derived hard negatives per positive. Training used 249 open Phase 7
queries and 1,245 pairs. Category and dialogue-template groups were held out;
missing-metadata matches and product/accessory confusions were included without
treating absent metadata as contradiction.

The frozen model hash is
`8a0fe765eb7925ea38c820a140f8eb518c3659b3596f0df1a309e70feffb4f05`.
Its ignored local asset lives at `artifacts/models/reranker_domain_v1` and can be
reproduced with `experiments/domain_reranker_train.py`. The runtime verifies the
model, tokenizer, revision, and every manifest checksum before loading it.

Open conditional ranking improved on both registered splits:

| Split | Base MRR | Candidate MRR | Base Top-10 | Candidate Top-10 |
|---|---:|---:|---:|---:|
| Category holdout | 0.726574 | 0.769911 | 0.925 | 0.950 |
| Template holdout | 0.760643 | 0.811368 | 0.950 | 0.975 |

The one-shot screening run is
`runs/phase11-domain-reranker-screening-20260830`. HitRate stayed `0.98125` and
TechnicalScore rose from the Phase 10 control's `0.865071` to `0.869003`, a
gain of `0.003932`. That misses the registered `0.005` threshold. Warm p95 rose
from `0.424832 s` to `0.638982 s` (about 50%), boundary MRR fell from
`0.483503` to `0.419501`, and buying MRR also declined slightly. D30, token
count, asset size, RSS, offline behavior, and zero-fallback behavior were
preserved.

The arm is rejected before confirmation. `configs/domain_reranker_v1.json` is
retained only as an auditable opt-in candidate; the base reranker remains the
release path.
