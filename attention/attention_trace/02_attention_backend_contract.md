# Attention Backend Contract

This note explains what an "attention backend" actually means in vLLM.

The short version is:

- a backend is not just a kernel name
- it is a class-level capability contract plus
- a metadata builder plus
- a runtime impl plus
- KV-cache shape/layout rules

The main file is `vllm/v1/attention/backend.py`.

## The Three Core Pieces

For normal v1 attention, the backend class must tell vLLM how to obtain:

- the backend class itself
- the metadata builder class
- the runtime impl class

The most important methods are:

- `get_name()`
- `get_impl_cls()`
- `get_builder_cls()`
- `get_kv_cache_shape(...)`

That gives you the three layers of the abstraction:

1. `AttentionBackend`: declarative capability and layout surface
2. `AttentionMetadataBuilder`: builds per-step/per-batch metadata for the backend
3. `AttentionImpl`: actually executes the attention path

## Registry And Registration

Backends are enumerated in `vllm/v1/attention/backends/registry.py`.

`AttentionBackendEnum` maps symbolic names to default class paths, for example:

- `FLASH_ATTN`
- `FLASHINFER`
- `TRITON_ATTN`
- `FLEX_ATTENTION`
- `CPU_ATTN`

Two important mechanics:

- `backend.get_class()` resolves the actual class path
- `register_backend(...)` can override an existing backend or register `AttentionBackendEnum.CUSTOM`

That means plugin-style integration can happen without changing the enum value to a brand-new built-in backend name, as long as you register an override or a custom backend path.

## Capability Methods

The backend contract separates simple capabilities from runtime implementation.

Important capability methods:

- `supports_head_size(...)` Whether the backend accepts the model's per-head hidden dimension.
- `supports_dtype(...)` Whether the backend supports the Q/K/V compute dtype for the layer.
- `supports_kv_cache_dtype(...)` Whether the backend supports the KV-cache storage format, including quantized cache types such as FP8-style modes.
- `supports_block_size(...)` Whether the backend can execute with the requested KV-cache block size, or with a framework block size compatible with its kernel granularity.
- `supports_sink()` Whether the backend supports attention-sink inputs used by StreamingLLM-style attention variants.
- `supports_mm_prefix()` Whether the backend supports decoder attention with multimodal prefix ranges that become bidirectional within selected spans.
- `supports_non_causal()` Whether the backend supports bidirectional decoder attention in the normal paged-KV decoder path.
- `supports_attn_type(...)` Whether the backend supports the requested attention mode such as decoder, encoder-only, encoder, or encoder-decoder.
- `supports_batch_invariance()` Whether the backend can run in vLLM's batch-invariant execution mode.
- `supports_kv_connector()` Whether the backend is compatible with KV-transfer / KV-connector flows where KV state may come from external transfer machinery. (LMCache compatibility)
- `supports_compute_capability(...)` Whether the backend is valid for the current hardware generation, usually expressed as a minimum or constrained GPU compute capability.

These are the methods that selection uses to decide whether the backend is even eligible for a given layer and configuration.

## Validation Entry Point

The main validation method is:

- `AttentionBackend.validate_configuration(...)`

This is the method platform selectors call during explicit selection and auto-selection. Uses the above mentioned capability methods to validate configuration.

It collects invalid reasons instead of returning only a boolean. That is why vLLM can explain selection failures such as:

- `head_size not supported`
- `kv_cache_dtype not supported`
- `attention type encoder_only not supported`
- `partial multimodal token full attention not supported`

## KV Cache Contract

Backends must define the logical KV-cache shape and may define the physical stride order separately.

Important methods:

- `get_kv_cache_shape(...)` Returns the logical tensor shape the backend expects for KV-cache storage, given the number of blocks, block size, KV heads, head size, and cache dtype.
- `get_kv_cache_block_dim(...)` Identifies which tensor dimension corresponds to the KV block index, so the worker can reason about block addressing even when layouts differ by backend.
- `get_kv_cache_stride_order(...)` Optionally tells vLLM how the physical memory order differs from the logical shape, so raw KV allocations can be viewed with the backend's preferred layout.
- `get_required_kv_cache_layout(...)` Lets a backend request a specific global KV layout mode, such as a particular head/block ordering, before runtime execution starts.

This is one of the most important parts of backend integration. A backend is not only choosing how to compute attention; it is also choosing how the KV cache must be shaped and interpreted.

Related block-size methods:

- `get_supported_kernel_block_sizes()` Declares the kernel block sizes or block-size granularities the backend can execute with.
- `supports_block_size(...)` Checks whether a requested framework KV block size is compatible with the backend's kernel requirements.
- `get_preferred_block_size(...)` Chooses the backend's preferred block size when the default KV-manager block size is not directly suitable.

These methods participate in both selection and worker-side kernel block-size negotiation.

## Metadata Builder Contract

`AttentionMetadataBuilder` is the bridge between common scheduler/runtime state and backend-specific execution metadata.

Important methods and properties:

- `build(...)` The main metadata construction entry point. It turns `CommonAttentionMetadata` plus batch-level context into the backend-specific metadata object that the impl will consume.
- `build_for_cudagraph_capture(...)` Builds metadata in a CUDA-graph-capture-friendly way. Backends can override this when graph capture needs stricter or more static metadata preparation.
- `build_for_drafting(...)` Builds metadata for speculative/draft-model execution, where the builder may choose a faster or differently structured metadata path than normal runtime building.
- `update_block_table(...)` Updates an existing metadata object with a new block table and slot mapping instead of rebuilding everything from scratch. This is useful when multiple KV-cache groups share nearly identical metadata.
- `get_cudagraph_support(...)` Reports the backend's CUDA-graph support level so the worker can decide which execution modes and capture strategies are valid.

The input to builders is `CommonAttentionMetadata`, which contains shared runtime state such as:

- query start locations
- sequence lengths
- block table tensor
- slot mapping
- DCP local sequence lengths
- optional encoder sequence lengths
- optional positions

Backends then turn that common state into metadata objects tailored to their own kernel interface.

This is why backend work in vLLM is usually two problems, not one:

- kernel execution
- metadata construction

## Runtime Impl Contract

`AttentionImplBase` and `AttentionImpl` define the runtime execution contract.

Important impl-level requirements:

- `forward(...)` is the actual runtime entry point
- `do_kv_cache_update(...)` may be required if KV update is split out of forward
- `supports_quant_query_input` controls whether the layer can quantize queries before dispatch

Important distributed capability flags on the impl:

- `can_return_lse_for_decode` Whether the implementation is capable of returning decode-time softmax log-sum-exp (LSE) alongside attention outputs. This is required in DCP-style distributed decode, where each rank computes only a partial attention result and the runtime needs LSE information to merge those partial results correctly.
- `supports_pcp` Whether the implementation supports Prefill Context Parallelism (PCP). This is checked after impl construction because PCP changes the runtime execution path and metadata requirements, not just the selector-time capability check.
- `supports_mtp_with_cp_non_trivial_interleave_size` Whether the implementation can handle MTP when context-parallel KV interleave size is greater than 1.

These are not selector-level capabilities. They are runtime requirements checked later by worker code when distributed attention is initialized.

## Distributed Requirement Nuance

For DCP specifically, the important requirement is:

- the impl must be able to return softmax LSE during decode

That requirement is enforced in `vllm/v1/worker/cp_utils.py`, not in the normal backend selector.

For PCP specifically, the important requirement is:

- the impl must explicitly support Prefill Context Parallelism

That requirement is also enforced in `vllm/v1/worker/cp_utils.py`, where vLLM checks the impl-level `supports_pcp` flag after construction.

So the two context-parallel modes differ in what they require from the impl:

- DCP requires decode-time softmax LSE support so partial attention results can be merged correctly across ranks
- PCP requires support for prefill context-parallel execution itself

So there are two different validation layers:

1. backend class capability validation during selection
2. impl-level distributed compatibility validation during runtime

This distinction matters a lot when designing a new backend.

## `AttentionLayer(Protocol)`

`AttentionImpl.forward(...)` receives an `AttentionLayer`-shaped object defined as a `Protocol` in `vllm/v1/attention/backend.py`.

This protocol describes the minimal layer surface that impls are allowed to depend on during execution, including:

- quantization scale tensors such as `_q_scale`, `_k_scale`, `_v_scale`
- host-side float mirrors such as `_q_scale_float`, `_k_scale_float`, `_v_scale_float`
- `_prob_scale`
- the layer's `forward(...)` shape contract

Why this matters:

- impls often need per-layer runtime state, especially quantization scales
- but they should not be tightly coupled to one concrete Python class
- the protocol makes that dependency explicit while keeping the impl interface generic

So `AttentionLayer(Protocol)` is the typed contract for "what the impl may read from the owning layer while executing attention."

## Derived / Wrapper Backends

vLLM also supports generating specialized backend subclasses at runtime via:

- `subclass_attention_backend(...)`
- `subclass_attention_backend_with_overrides(...)`

These are used for wrapper variants such as:

- encoder-only attention
- cross attention
- static sink attention
- chunked local attention

For the generic architecture, the important point is that a "backend class" may be a synthesized subclass, not only one of the raw registry classes.

## Practical Checklist For A New Backend

At minimum, a new generic backend needs to provide:

- a registered backend class
- an impl class
- a metadata builder class
- KV-cache shape rules
- block-size support rules
- capability methods that correctly describe what the backend can do

And if the backend is meant to work in distributed decode or context-parallel setups, it also needs the corresponding impl-level distributed features.

## Best Files To Read Next

- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/backends/flex_attention.py`
