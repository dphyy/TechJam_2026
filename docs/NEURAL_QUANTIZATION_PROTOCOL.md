# MiniLM Dynamic Quantization Protocol

## Bounded candidate

Phase 6 evaluates one compression candidate: PyTorch eager dynamic int8
quantization of `Linear` layers in the selected local MiniLM, using the QNNPACK
backend available on the submission Mac. Query, D30 candidates, serializer,
maximum sequence length, four CPU threads, batch size 16, and neural fusion
weight remain fixed.

This is a feasibility boundary before runtime integration. It measures two
warmups and 20 repetitions in separate fresh processes, with exact input hashes,
logits, final fused ranking, p50/p95/max, RSS, cold start, token count, and a
serialized state-dict size estimate.

The candidate must show a 25% resource win without creating an obviously unsafe
or unsupported deployment path. A feasibility failure does not justify opening
the final outcome split.

## Result

| Arm | p50 | p95 | Cold start | RSS | Serialized bytes |
|---|---:|---:|---:|---:|---:|
| Float32 control | 0.183s | 0.186s | 1.728s | 660,307,968 | 90,870,598 |
| Dynamic int8 QNNPACK | 0.405s | 0.408s | 1.861s | 727,842,816 | 58,631,297 |

The serialized state estimate is 35.48% smaller, but p95 regresses 119.45% and
RSS rises 10.23%. Prompt tokens are identical. The fused ranking changes and
maximum absolute logit drift is `0.651466` on the fixed D30 query.

The installed PyTorch version also warns that eager `torch.ao.quantization` is
deprecated and that QNNPACK currently ignores `reduce_range`. The serialized
state estimate is not a standalone pinned asset: the feasibility implementation
first loads the float32 model and then quantizes it in memory, so it would not
deliver the measured disk reduction in the actual submission as written.

Phase 6 is rejected before runtime integration. No configuration or selected
behavior changes, no end-to-end outcome set is opened, and final remains sealed.
