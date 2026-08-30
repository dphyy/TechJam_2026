# Phase 15 Intent Classifier Diagnostics

Recorded 29 August 2026. This milestone implements dataset and classifier
diagnostics only. It does not change `mercury/intent.py`, runtime routing, retrieval,
dialogue policy, or `configs/selected.json`.

## Frozen protocol

- The committed source corpus contains 120 independently authored utterances in
  60 isolation groups, balanced across `buying`, `browsing`, and `mixed`.
- Each group fixes an author ID, paraphrase family, intent card, and product family.
  Both utterances in a group remain in the same split.
- The deterministic seed `20260829` assigns exactly 14/3/3 groups per class to
  train/validation/sealed test: 84/18/18 utterances, or 70/15/15 percent.
- Authored slices include direct, indirect, mixed, correction, override, vague,
  conflicting, and out-of-vocabulary language. No public-set or `unseen-v1` row,
  target, reply, or model prediction was used to author or split the corpus.
- Coefficients are fitted on train. Regularization, temperature calibration, and
  confidence abstention are selected on validation. The sealed test has one
  create-only consumption marker and was opened once after all hashes were frozen.
- The semantic encoder is the already declared local
  `BAAI/bge-small-en-v1.5` revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; there is no network inference or
  paid service.

Important frozen hashes:

```text
authored source    f050ace2ca912c2a453ee54e4f76dfaf9110daff8fbc6b2fba8cd7e30503e911
train              ee65f88a83b83543c476aa773f22704bc335907c44d0bb1491193fe3e804711b
validation         1c51f3dfa8d9a475fd577f6fec50a0231944e92946776ee4ef9a002cfd26c654
sealed test        5ca67d108bc37798eca33109a0b483cb68832846f30fa5b7dbb694636001c871
model/calibration  11b4e3b4b708eab5c0859b923e19bc186995343b319b30c4f4f0255c3adbe9df
```

## Sealed-test results

The balanced sealed set has 18 utterances, six per class. Macro F1 includes an
abstention as a missed prediction. Latency is warm batch time on the recorded local
arm64 environment; semantic and hybrid rows include the shared encoding cost.

| Baseline | Macro F1 | Log loss | Brier | ECE | Abstain | Covered accuracy | ms/row |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current rules | 0.420513 | 1.096749 | 0.650276 | 0.262521 | 0.000000 | 0.500000 | 0.198005 |
| Structural linear | 0.487179 | 1.060525 | 0.641046 | 0.194882 | 0.166667 | 0.600000 | 0.097819 |
| Semantic linear | **0.487879** | **0.969522** | **0.603628** | 0.216212 | 0.166667 | 0.533333 | 2.369160 |
| Hybrid linear | 0.402694 | 0.967201 | 0.610567 | **0.167681** | 0.333333 | 0.500000 | 2.466979 |

Per-class recall exposes the main failure rather than the aggregate alone:

| Baseline | Buying | Browsing | Mixed |
|---|---:|---:|---:|
| Current rules | 1.000000 | 0.000000 | 0.500000 |
| Structural linear | 1.000000 | 0.500000 | 0.000000 |
| Semantic linear | 0.500000 | 0.500000 | 0.333333 |
| Hybrid linear | 0.333333 | 0.333333 | 0.333333 |

The semantic model improves sealed macro F1 by `+0.067366` over current rules and
has better log loss and Brier score, but this is only 18 sealed examples. It does
not establish downstream value. Structural and hybrid confidence abstention also
gave up substantial coverage, and mixed/conflicting slices remain particularly
weak. The hybrid did not improve on semantic-only behavior, so the current state
features do not yet add robust complementary signal.

## Decision

Do not promote any learned classifier. Keep the selected D30 release and current
runtime unchanged. The diagnostic result is useful evidence that semantic language
coverage can help, while also showing that the present corpus is too small and the
`mixed` boundary is not learned reliably enough.

Any next attempt should be a newly registered experiment, not sealed-test tuning:

1. Add more author/user groups, with extra mixed, correction, override, and
   conflicting examples, then create a new sealed version before inference.
2. Audit label agreement with at least one independent annotator and report
   disagreement rather than silently resolving ambiguous turns.
3. Improve structural state-transition features on development only, particularly
   object-choice uncertainty and explicit relaxation versus committed correction.
4. If a classifier passes a larger intent-only gate, freeze it and evaluate one
   joint retrieval/dialogue candidate on a new target/user-disjoint downstream
   pack. Intent accuracy alone cannot change routing.

The sealed evaluation emitted a macOS Accelerate floating-point warning while
producing finite logits. After the one permitted evaluation, inference gained the
same explicit finite-output validation already used during fitting. This
non-semantic hardening did not reopen or alter the recorded sealed result.

## Reproduction commands

Use new output paths. A prepared split intentionally refuses overwrite, and its
sealed test intentionally refuses a second evaluation.

```bash
python -m experiments.prepare_models --model embedding --download
python -m experiments.intent_dataset --output artifacts/intent-v1
python -m experiments.intent_diagnostics fit \
  --data-root artifacts/intent-v1 \
  --output artifacts/intent-v1-fit
python -m experiments.intent_diagnostics evaluate-sealed \
  --freeze artifacts/intent-v1-fit/model-freeze.json \
  --output runs/phase15-intent-sealed
```
