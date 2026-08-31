> Optional research assets only. The current lexical search with guarded paging loads no model and needs no weights, embeddings, API key, NumPy, or Torch. The model selections below describe historical experiments. See [setup](SETUP.md).

# Optional local models

The sparse agent has no neural dependencies. Neural experiments use these immutable, English-language public models; no remote inference service is used.

| Role | Model and revision | License | Weight size |
|---|---|---|---|
| Dense retrieval | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | MIT | About 133 MB |
| Semantic reranking | [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) | Apache-2.0 | About 91 MB |

Weights are safetensors only; remote code execution is disabled. Acquisition is explicit and separate from runtime. Asset manifests include model revisions and file hashes. BGE must retain its CLS pooling configuration, modules metadata, and normalization; missing metadata must not silently create mean pooling.

The query-only BGE retrieval prefix is `Represent this sentence for searching relevant passages: `. Product text is not prefixed. Reranking uses raw logits, not calibrated probabilities. Both models were developed outside this shopping task; measure their actual value and limitations rather than assuming a newer model is better.

Optional neural/research dependencies are pinned in
[requirements-research.txt](../requirements-research.txt). The public runtime's
[requirements.txt](../requirements.txt) has no third-party requirements; optional
lint tooling is in [requirements-dev.txt](../requirements-dev.txt). Current public
verification uses Python 3.13.5 and does not load either model. Legacy model runs
need their recorded source, assets and environment; historical inference checks
are not evidence that a newly installed environment has been validated. See
[setup](SETUP.md) and [research comparisons](RESEARCH_INDEX.md).
