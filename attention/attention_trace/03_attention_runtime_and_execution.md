# Attention Runtime And Execution

This note maps the runtime path after a backend has been selected.

The important distinction is:

- selection chooses a backend class
- runtime constructs groups, metadata builders, KV-cache views, and finally executes the impl

The main files are:

- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/utils.py`
- `vllm/v1/worker/gpu/attn_utils.py`

## Layer-Time Construction

Each `Attention` layer is constructed in `vllm/model_executor/layers/attention/attention.py`.

During `Attention.__init__(...)` in `vllm/model_executor/layers/attention/attention.py`, the layer computes:

- `sliding_window`
- `kv_cache_dtype` from `cache_config`
- whether KV scales should be calculated
- `num_heads`, `num_kv_heads`, `head_size`
- `has_sink` in `extra_impl_args`
- `use_mm_prefix` from `model_config`

If `attn_backend` was not injected directly, the layer calls `get_attn_backend(...)` from `vllm/v1/attention/selector.py` and stores the resulting backend class.

Then it instantiates the impl:

- `impl_cls = self.attn_backend.get_impl_cls()`
- `self.impl = impl_cls(...)`

What these two lines mean:

- `self.attn_backend` is still just the selected backend class, for example a class such as `FlashAttentionBackend` or `TritonAttentionBackend`
- `get_impl_cls()` asks that backend class which concrete runtime implementation class should do the actual work, for example `FlashAttentionImpl`
- `impl_cls(...)` then constructs that runtime object using the layer's actual parameters such as `num_heads`, `head_size`, `num_kv_heads`, `sliding_window`, `kv_cache_dtype`, `attn_type`, and any backend-specific extra arguments

Note for turboquant: how we can map `get_impl_cls()` and `impl_cls(...)`

So the backend class is the declarative/configuration side, while `self.impl` is the stateful execution-side object that will later receive:

- query/key/value tensors
- KV cache tensor
- backend-specific attention metadata

and run `forward(...)` or KV-cache update logic.

So the layer owns the runtime impl instance, not just the backend class.

## Runtime Grouping

The worker later regroups attention layers rather than treating every layer fully independently.

In `GPUModelRunner.initialize_attn_backend(...)` in `vllm/v1/worker/gpu_model_runner.py`, layers are grouped by:

- backend class
- KV-cache spec
- per-rank query-head count

Those groups become `AttentionGroup` objects from `vllm/v1/worker/utils.py`.

The relevant code for this section is:

- local `AttentionGroupKey` inside `GPUModelRunner.initialize_attn_backend(...)` in `vllm/v1/worker/gpu_model_runner.py`
- local helper `get_attn_backends_for_group(...)` in the same method
- local helper `create_attn_groups(...)` in the same method
- `AttentionGroup.create_metadata_builders(...)` in `vllm/v1/worker/utils.py`

What the grouping dimensions actually mean:

- backend class: layers using different backends cannot share metadata builders or execution assumptions, because each backend expects different metadata layouts, scratch buffers, and runtime behavior.
- KV-cache spec: layers that differ in KV-cache structure also cannot safely share builders. The KV-cache spec captures properties such as block size, KV head count, head size, cache dtype family, and whether the layer is using a full-attention or sliding-window style spec.
- per-rank query-head count: even if the backend and KV-cache spec match, layers may still need separate groups if the local number of Q heads differs on this rank. The code comment in `initialize_attn_backend(...)` explains why: some builders size internal scratch buffers directly from `num_heads_q`, so mixing layers with different local Q-head counts would make that builder state invalid.

There is also one important preprocessing step in `get_attn_backends_for_group(...)`:

- if a layer is eligible for fast-prefill KV sharing, the original backend can be wrapped with `create_fast_prefill_custom_backend(...)`

Another subtle detail from the code:

- grouping uses `attn_backend.full_cls_name()` rather than the raw class object

That is defensive against dynamically generated backend subclasses. Two layers may conceptually use the same backend family but still hold different Python class objects if the backend was synthesized or wrapped at runtime.

The flow inside `initialize_attn_backend(...)` is:

1. iterate over each KV-cache group in `kv_cache_config.kv_cache_groups`
2. collect the relevant layers for that KV-cache group
3. compute an `AttentionGroupKey` for each layer
4. bucket layer names by that key
5. convert each bucket into an `AttentionGroup`
6. later, create metadata builders per group in `initialize_metadata_builders(...)`

Why this matters:

- metadata builders are reusable only when the grouped layers are truly homogeneous from the builder's point of view
- kernel block-size negotiation happens per KV-cache group, but builder construction and metadata scratch ownership happen per attention group
- CUDA-graph support checks also consume the backend sets discovered during this grouping phase

So an `AttentionGroup` is best thought of as:

- one homogeneous runtime-attention bucket inside a KV-cache group

not merely:

- a convenient list of layers

## Metadata Builders

After groups are created, `GPUModelRunner.initialize_metadata_builders(...)` in `vllm/v1/worker/gpu_model_runner.py` creates the backend-specific builders for each group.

The builder constructor receives:

- the group's KV-cache spec
- layer names in that group
- `VllmConfig`
- device

This is where vLLM turns "we selected backend X" into "we now have a concrete object that knows how to build runtime metadata for backend X."

## Kernel Block-Size Negotiation

Kernel block size is negotiated separately from the framework KV block size.

The worker-side flow is:

1. derive KV-cache groups
2. derive attention groups within each KV-cache group
3. inspect all attention-group backends for that KV-cache group
4. call `prepare_kernel_block_sizes(...)`
5. choose a common kernel block size supported by that KV-cache group

The key methods are in `vllm/v1/worker/utils.py`:

- `select_common_block_size(...)`
- `prepare_kernel_block_sizes(...)`

Important distinction:

- an `AttentionGroup` is homogeneous by construction, so it has one backend
- a `KV-cache group` can still contain multiple `AttentionGroup`s

That is why `prepare_kernel_block_sizes(...)` looks at all group backends for a single KV-cache group. The KV cache is shared at the KV-cache-group level, so the selected kernel block size must be compatible with every attention backend that will use that shared KV layout.

This is where virtual block splitting becomes concrete:

- the KV manager may own larger logical blocks
- the backend may execute using smaller kernel blocks

## KV Cache View Construction

The KV cache manager allocates raw memory, but the backend determines the shaped view over that memory.

Worker code reshapes raw tensors by calling backend methods defined in `vllm/v1/attention/backend.py`:

- `backend.get_kv_cache_shape(...)`
- optionally `backend.get_kv_cache_stride_order(...)`

This happens during KV-cache reshaping in `vllm/v1/worker/gpu/attn_utils.py`.

So the execution path depends on backend-specific KV layout even before the first forward call.

## Forward Context

At runtime, the model runner installs attention metadata in a forward context. The shared metadata container itself is `CommonAttentionMetadata` from `vllm/v1/attention/backend.py`.

What `ForwardContext` is:

- `ForwardContext` is a per-forward-pass runtime context object defined in `vllm/forward_context.py`
- it is installed temporarily around one model execution using `set_forward_context(...)` from `vllm/forward_context.py`
- during that active window, internal layers and custom ops can call `get_forward_context()` to fetch runtime state without having all of that state threaded through every Python `forward(...)` signature

So `ForwardContext` is not long-lived model state. It is ephemeral execution state for the current forward pass.

Why vLLM uses it:

- the runtime needs to make per-step state such as attention metadata, slot mappings, CUDA-graph mode, and other execution descriptors visible to many internal components
- passing all of that explicitly through every intermediate layer call would make the model interfaces much wider and harder to maintain

Representative fields in `ForwardContext` include:

- `attn_metadata`
- `slot_mapping`
- `no_compile_layers`
- `dp_metadata`
- `cudagraph_runtime_mode`
- `batch_descriptor`
- `ubatch_slices`
- `skip_compiled`

`Attention.forward(...)` does not receive `attn_metadata` directly as a Python argument from the model definition. Instead it looks it up from:

- `get_forward_context()` from `vllm/forward_context.py`

The helper `get_attention_context(...)` in `vllm/model_executor/layers/attention/attention.py` extracts:

- the layer-specific attention metadata
- the attention layer instance
- the KV cache tensor
- the slot mapping

This is a key architectural point: the attention layer reads runtime execution state from a shared forward context.

## Custom-Op Boundary

The generic attention path is intentionally wrapped in custom ops.

The important registered ops in `vllm/model_executor/layers/attention/attention.py` are:

- `unified_kv_cache_update`
- `unified_attention_with_output`
- `maybe_calc_kv_scales`

What each one does:

- `maybe_calc_kv_scales` Runs optional KV-scale calculation for layers that need runtime KV quantization scale updates. It resolves the layer object from `ForwardContext.no_compile_layers`, checks `self.calculate_kv_scales`, and if enabled calls the layer's `calc_kv_scales(...)` logic on the current query/key/value tensors.
- `unified_kv_cache_update` Performs the KV-cache write/update path separately from the main attention compute when the backend does not include KV update inside `forward(...)`. It resolves the current layer context, gets the layer-specific slot mapping, and calls `attn_layer.impl.do_kv_cache_update(...)`. It also returns a dummy tensor so the later attention op can depend on it and preserve ordering under torch.compile`.
- `unified_attention_with_output` Is the main execution entry point. It resolves the active attention metadata, layer object, and KV cache from `ForwardContext`, then calls `self.impl.forward(...)` with the current query/key/value tensors and output buffers.

Why the boundary is drawn here:

- above this point, vLLM still works in terms of generic layer logic such as reshaping Q/K/V, setting up output buffers, and reading the current forward context
- at this point, vLLM has resolved all runtime state needed by the backend: metadata, KV cache tensor, slot mapping, and the concrete impl object
- below this point, execution becomes backend-specific and may involve custom kernels, backend-specific metadata layouts, fused KV updates, quantized storage rules, or distributed attention combine logic

So the custom-op boundary is effectively the handoff from:

- generic vLLM layer/runtime orchestration

to:

- backend-specific attention execution

This boundary is also useful for compilation/runtime reasons:

- it keeps the Python-visible layer API small
- it gives `torch.compile` a stable opaque op boundary on platforms where vLLM wants to control attention execution as one unit
- it lets vLLM preserve ordering between KV-cache update and attention compute through explicit dummy dependencies when needed

The forward path is:

1. reshape query/key/value outside the custom op
2. optionally update KV cache
3. invoke `unified_attention_with_output`
4. inside that op, call `self.impl.forward(...)`

Small note on step 2:

- this step only happens as a separate custom-op call when the backend does not include KV-cache update inside its main `forward(...)`
- the layer checks `self.attn_backend.forward_includes_kv_cache_update`
- if that flag is `False`, vLLM calls `unified_kv_cache_update` first
- if that flag is `True`, the backend's `self.impl.forward(...)` is expected to perform the KV write/update itself as part of the main attention execution path

So the backend impl executes behind the unified attention custom-op boundary.

This is the place where "backend selected" becomes "backend actually runs."

## Direct Call vs Opaque Op

`Attention` also has a platform-dependent execution mode:

- opaque custom-op path for CUDA-like and CPU platforms
- direct-call mode for platforms where vLLM does not wrap attention as one opaque op

What these mean:

- opaque custom-op path vLLM exposes attention to PyTorch/`torch.compile` as a registered custom op boundary such as `torch.ops.vllm.unified_attention_with_output`. From the compiler's point of view, the internals of attention are treated as one opaque unit rather than as a large Python/Torch graph to inspect and rewrite.
- direct-call mode vLLM skips the opaque custom-op wrapping and directly calls the Python-side attention helper / impl path. In this mode, attention is not intentionally hidden behind one custom-op boundary by vLLM.

This is controlled by:

- `current_platform.opaque_attention_op()` from `vllm/platforms/interface.py`

Why vLLM prefers the opaque-op mode on CUDA-like platforms:

- it gives vLLM tighter control over the execution boundary
- it reduces the amount of attention internals that `torch.compile` needs to reason about
- it makes ordering and side-effect handling around KV-cache update easier to manage

For most backend-plugin work on CUDA, the opaque custom-op path is the one that matters.

## Prefill vs Decode

The same layer/impl supports both prefill and decode, but metadata and backend behavior differ.

At a high level:

- prefill processes many prompt tokens together
- decode processes new tokens against cached KV

The impl sees this through backend-specific metadata objects built by `AttentionMetadataBuilder` subclasses from `vllm/v1/attention/backend.py`, rather than through a totally different public API.

Some practical differences between the two:

- query shape and query lengths In prefill, a request may contribute many query tokens in one step, so `query_start_loc`, `max_query_len`, and per-request query lengths can be much larger than 1. In decode, the common case is one new token per request, so query lengths are usually 1 or very small.
- KV usage In prefill, the step is often building up a large amount of KV for prompt tokens while also attending over the prompt with causal masking. In decode, the new query tokens primarily attend against already cached historical KV plus the just-added current-step tokens.
- sequence-length interpretation `seq_lens` in metadata represents the total usable context length for each request at that step. During decode this usually means "large cached history plus a tiny new query," while during prefill it can mean "the prompt chunk currently being computed."
- backend scheduling choices Some backends choose different internal paths, kernels, or metadata layouts depending on whether the batch looks like prefill, decode, or mixed execution. That distinction is inferred from metadata rather than from a separate public method.
- distributed implications DCP/PCP-style distributed constraints tend to matter differently across the two phases: DCP is specifically about decode-time context sharding, while PCP is specifically about prefill-time context sharding.

Another useful way to think about it:

- prefill is usually the "wide query" phase
- decode is usually the "tiny query against large cached context" phase

That difference is one of the main reasons metadata builders are so important. The public layer API stays the same, but the backend sees a very different execution situation through metadata.

## Where Runtime Specialization Happens

Several runtime features are decided here rather than in the selector:

- CUDA-graph support level
- group-wise metadata builder reuse
- DCP/PCP compatibility checks
- `mm_prefix` metadata injection
- sliding-window runtime mask configuration

This is why a backend that looks "valid" at selection time can still fail later if its impl or metadata builder does not satisfy runtime expectations.

## Best Files To Read Next

- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/utils.py`
