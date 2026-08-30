# Optional local models

The sparse agent has no neural dependencies. Neural experiments use these immutable, English-language public models; no remote inference service is used.

| Role | Model and revision | License | Weight size |
|---|---|---|---|
| Dense retrieval | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | MIT | About 133 MB |
| Semantic reranking | [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) | Apache-2.0 | About 91 MB |

Weights are safetensors only; remote code execution is disabled. Acquisition is explicit and separate from runtime. Asset manifests include model revisions and file hashes. BGE must retain its CLS pooling configuration, modules metadata, and normalization; missing metadata must not silently create mean pooling.

The query-only BGE retrieval prefix is `Represent this sentence for searching relevant passages: `. Product text is not prefixed. Reranking uses raw logits, not calibrated probabilities. Both models were developed outside this shopping task; measure their actual value and limitations rather than assuming a newer model is better.

Pinned library versions are consolidated in `requirements.txt`. It was installed and checked in a fresh CPython 3.12.12 macOS arm64 environment. Both models completed real inference with OS-level network access denied and provider credentials removed. Source APIs: [SentenceTransformer v5.1.2](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/sentence_transformers/SentenceTransformer.py), [CrossEncoder v5.1.2](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/sentence_transformers/cross_encoder/CrossEncoder.py), and [offline mode](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline).
