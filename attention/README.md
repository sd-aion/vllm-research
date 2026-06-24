# vLLM Attention Research Index

This directory maps the vLLM v1 attention architecture from backend selection to kernel execution. It begins with the backend-independent architecture and then traces FlashAttention, Triton attention, and TurboQuant as concrete implementations.

The source checkout used by these notes is `/home/ubuntu/vllm`.

## Generic Attention Architecture

Read the generic trace before a backend-specific deep dive:

| Part | Document | Coverage |
| ---: | --- | --- |
| 1 | [Attention entry and selection](attention_trace/01_attention_entry_and_selection.md) | Launch arguments, `AttentionConfig`, explicit backend choice, automatic platform selection, and selection-time validation. |
| 2 | [Attention backend contract](attention_trace/02_attention_backend_contract.md) | Backend capabilities, KV-cache layout methods, metadata builders, runtime implementation methods, output buffers, CUDA graph support, DCP, and PCP requirements. |
| 3 | [Attention runtime and execution](attention_trace/03_attention_runtime_and_execution.md) | Layer construction, attention grouping, metadata-builder initialization, forward context, custom-op boundaries, KV-cache update, and backend execution. |
| 4 | [Distributed attention](attention_trace/04_attention_distributed.md) | TP, PP, DP, SP, PCP, DCP, local head ownership, communication requirements, and distributed kernel constraints. |
| 5 | [Sliding-window and multimodal-prefix attention](attention_trace/05_attention_sliding_window_and_mm_prefix.md) | Local attention windows, multimodal bidirectional prefix ranges, metadata, masking, and backend requirements. |
| 6 | [Multimodal encoder attention](attention_trace/06_multimodal_encoder_attention.md) | The separate encoder-attention path used by ViT-style multimodal components. |

The [attention backend legend reference](attention_backend_legend_explained.md) explains every column in `vllm/docs/design/attention_backends.md`, including dtypes, KV dtypes, block and head sizes, sinks, non-causal attention, sparse attention, multimodal prefix support, DCP, attention types, and compute capability.

## Backend Deep Dives

### FlashAttention

Start with the [FlashAttention trace README](flash_attention_trace/README.md).

This nine-part trace covers:

- Selection and FA2/FA3/FA4 version resolution
- Backend capabilities and the standard-attention feature table
- Metadata construction and CUDA graph behavior
- KV-cache shape, layout, block tables, and slot mappings
- First prefill, continuation prefill, and decode
- FlashAttention library entry points and vLLM helper kernels
- TP, DCP, PCP, and distributed state merging
- Sliding window, ALiBi, sinks, non-causal attention, multimodal prefixes, and cascade attention

Use this trace as the model for a backend that delegates its main attention computation to an external kernel library while vLLM owns integration, cache updates, and distributed orchestration.

### Triton Attention

Start with the [Triton attention trace README](triton_attention_trace/README.md).

This fourteen-part trace covers:

- Backend selection, capabilities, cache shape, and metadata
- Standard decoder prefill, decode, and mixed-batch dispatch
- KV-cache update and quantized-cache store kernels
- The unified-attention wrapper and `kernel_unified_attention(...)`
- 2D and segmented 3D decode, split reduction, and launch thresholds
- Query-block mapping, cumulative sequence offsets, and helper algorithms
- Direct packed-Q/K/V encoder attention
- Adjacent paged-decode kernels, feature limitations, tests, and debugging

Use this trace when the backend and kernels are implemented inside vLLM and you need to understand every launch argument, tile, mask, pointer, and reduction path.

### TurboQuant

Start with the [TurboQuant trace README](turboquant_attention_trace/README.md).

This eighteen-part trace covers:

- Selection through `turboquant_*` KV-cache dtypes
- The paper-style key/value quantization and inference algorithms
- Combined uint8 KV-cache slots and packed representations
- Backend, implementation, metadata, and CUDA graph contracts
- Cache addressing and FP8, MSE-key, and affine-value store kernels
- First-chunk prefill, synthetic-decode continuation, and large continuation prefill
- Query rotation, split-KV decode, online softmax, and stage-two reduction
- Full cache dequantization, distributed limitations, tests, and debugging

Use this trace as an example of an attention backend that changes the physical cache representation and uses different computation strategies for prefill, decode, and continuation prefill.

## Choosing What To Read

| Goal | Recommended documents |
| --- | --- |
| Understand backend selection | Generic parts 1 and 2, then the selected backend's first two chapters. |
| Understand layer construction and calls | Generic part 3, then the backend's metadata and forward-path chapters. |
| Understand block tables or slot mappings | Generic part 3, the backend's cache chapter, and the KV-cache runtime trace linked from the repository root. |
| Add a new Triton kernel | Generic parts 2 and 3, then Triton chapters 5 through 11. |
| Add a custom KV-cache format | Generic parts 2 and 3, TurboQuant chapters 2 through 10, and the KV-cache integration trace. |
| Support multi-GPU attention | Generic part 4, then the selected backend's distributed chapter. |
| Support sliding window or multimodal prefixes | Generic part 5, then the selected backend's feature-variant chapter. |
| Debug prefill versus decode | Generic part 3, then the selected backend's metadata, forward, and kernel chapters. |

## Shared Runtime Model

Across the traced backends, the recurring execution model is:

1. Launch configuration requests a backend or allows platform auto-selection.
2. The selector validates static backend capabilities.
3. Each model attention layer constructs the backend's `AttentionImpl`.
4. The worker groups compatible layers and constructs backend-specific metadata builders.
5. The scheduler allocates logical KV-cache blocks; the worker owns physical cache tensors.
6. Each scheduler step produces sequence lengths, query offsets, block tables, and slot mappings.
7. The model runner installs this state in the forward context.
8. The shared `Attention` custom op updates KV cache when required and calls the backend implementation.
9. The backend dispatches prefill, decode, mixed-batch, or specialized attention behavior.
10. Distributed wrappers merge partial attention states when TP, PCP, or DCP requires communication.

## Main Source Anchors

- `vllm/engine/arg_utils.py`
- `vllm/config/attention.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/selector.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/platforms/cuda.py`
- `vllm/model_executor/layers/attention/attention.py`
- `vllm/forward_context.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/utils.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/kv_cache_interface.py`

## Scope

The generic trace focuses on v1 generic attention. The collection does not comprehensively cover MLA-specific backends, Mamba state-space layers, ROCm-only attention implementations, or legacy v0 paths.
