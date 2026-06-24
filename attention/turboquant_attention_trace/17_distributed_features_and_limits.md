# Distributed Features And Limits

TurboQuant contains no collectives. It operates on tensors and cache pages already local to the current worker rank.

## Tensor Parallelism

`ModelConfig.get_num_attention_heads(...)` and `get_num_kv_heads(...)` in `vllm/config/model.py` derive per-rank head counts from tensor-parallel size before constructing attention layers.

`TurboQuantAttentionImpl.num_heads` and `num_kv_heads` are therefore local counts. Each TP rank:

- Receives its local Q/K/V head shards.
- Allocates packed cache only for local KV heads.
- Builds the same `D x D` Hadamard transform for each local head vector.
- Runs store and decode only for local heads.
- Produces its local attention-head output shard for the surrounding tensor-parallel projection machinery.

No TurboQuant kernel communicates across TP ranks because attention heads are the sharding boundary.

MQA requires care in generic TP setup because there may be fewer KV heads than TP ranks. vLLM's model configuration returns at least one local KV head and may replicate KV heads where necessary; TurboQuant consumes the resulting local head assignment rather than implementing replication itself.

## Pipeline Parallelism

Pipeline parallelism places different model layers on different ranks.

Each rank constructs TurboQuant state and cache only for its resident attention layers. Pipeline activation transfers happen outside TurboQuant, and there is no cross-stage KV-cache operation in these kernels.

## Data Parallelism

Each data-parallel replica owns its requests, block tables, packed cache pages, quantization state, and workspaces.

TurboQuant does not synchronize compressed cache contents across DP replicas.

## Decode Context Parallelism

TurboQuant does not support DCP.

`TurboQuantAttentionImpl` inherits `can_return_lse_for_decode = False`, so `need_to_return_lse_for_decode` remains false even when a DCP group exists.

`check_attention_cp_compatibility(...)` in `vllm/v1/worker/cp_utils.py` requires decode implementations to return LSE when `decode_context_parallel_size > 1` and rejects TurboQuant.

The local stage-two kernel computes LSE only to merge local KV splits. Exposing and combining rank-partitioned attention states would require additional runtime and communication support.

## Prefill Context Parallelism

TurboQuant does not set `supports_pcp = True`, so PCP is rejected when `prefill_context_parallel_size > 1`.

PCP would require prefill kernels to operate on rank-local prompt partitions and participate in the distributed combination protocol. TurboQuant's first-chunk FlashAttention/SDPA path assumes the required prompt K/V are locally available, while its continuation logic iterates complete request metadata locally.

## CUDA Graphs

The metadata builder advertises `AttentionCGSupport.UNIFORM_BATCH`, not unconditional support.

The fixed `tq_max_kv_splits_for_cuda_graph` stabilizes decode grids and buffers. The workspace manager provides stable shared storage and prevents layer-count multiplication of decode and dequant scratch allocations.

Arbitrary mixed query lengths and Python per-request continuation-prefill control flow are not equivalent to a fully graphable uniform decode batch.

## Unsupported Table Features

- Attention sinks are unsupported because no sink logit participates in stage-one softmax initialization.
- Non-causal decoder attention is unsupported because prefill is causal and decode exposes only prefix positions through `seq_lens`.
- Multimodal prefix full attention is unsupported because metadata has no MM ranges and kernels have no bidirectional-range mask override.
- Encoder, encoder-only, and encoder-decoder attention types are rejected by `supports_attn_type(...)`.
- Generic per-head KV quantization scales are unsupported because scales and norms are encoded inside each TurboQuant slot.

## Additional Runtime Limitations

`alibi_slopes`, `logits_soft_cap`, and `sliding_window` are accepted by the common implementation constructor but are not applied by the TurboQuant kernels.

The selector table reports any positive head size, but the MSE path's `_build_hadamard(...)` doubles a Sylvester matrix until its size is at least `D`. For a non-power-of-two `D`, that produces a matrix larger than `D x D`, which is incompatible with the current `[..., D] @ PiT` multiplication. Runtime tests cover Hadamard execution for 64, 128, and 256, so MSE presets should presently be treated as requiring power-of-two head dimensions unless this construction is generalized.

FP8-key decode does not use the Hadamard multiplication, but `_ensure_on_device(...)` still builds the matrix. Any claim of arbitrary runtime head-size support should therefore be validated on the target preset and path rather than inferred solely from `supports_head_size(...)`.

The backend does not apply softcap or ALiBi despite accepting constructor arguments, so models requiring those score transformations are not semantically supported by the current implementation.

## KV Connectors

The backend inherits generic KV-connector support, but a connector must transfer opaque packed uint8 pages together with the matching TurboQuant preset and cache spec.

Converting only the tensor shape or treating the bytes as standard K/V elements would corrupt both key and value interpretation.
