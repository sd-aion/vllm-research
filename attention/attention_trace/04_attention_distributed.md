# Attention In Distributed Setups

This note covers how the generic attention path changes under multi-GPU and distributed execution.

The short version is:

- some parallel modes mostly affect scheduling or model partitioning
- some directly change attention tensor ownership, metadata, or kernel requirements
- the most important generic attention modes are TP, DCP, and PCP

## Parallelism Map

For generic attention, the current modes matter in different ways:

- `tensor_parallel_size`: directly changes local head ownership and KV head ownership.
- `decode_context_parallel_size`: directly changes how decode KV is sharded and what the kernel must return.
- `prefill_context_parallel_size`: directly changes whether impls must support PCP.
- `pipeline_parallel_size`: splits layers across stages, but does not usually change the backend contract of a given attention layer.
- `data_parallel_size`: replicates execution across ranks and usually does not change a backend's per-layer attention interface.
- `expert parallel` / `MoE-related modes`: mostly affect MoE communication, not the generic attention backend contract.
- `sequence-parallel` compilation paths: matter for compiled scheduling and size constraints, but are not a primary backend selector input in the generic v1 attention path.

The main config is `vllm/config/parallel.py`.

The most attention-relevant fields there are:

- `tensor_parallel_size` Number of tensor-parallel ranks. For generic attention this primarily decides how attention heads and KV heads are sharded across ranks.
- `pipeline_parallel_size` Number of pipeline stages. This partitions model layers across stages but usually does not change the internal attention backend contract of a single layer.
- `data_parallel_size` Number of data-parallel replicas. This mostly replicates model execution across ranks and is usually more of a deployment/runtime topology concern than an attention-kernel-interface concern.
- `prefill_context_parallel_size` Number of context-parallel groups used during prefill. This is the PCP knob and matters only if the attention impl supports prefill context partitioning.
- `decode_context_parallel_size` Number of context-parallel groups used during decode. This is the DCP knob and determines whether decode KV/cache context is sharded across ranks.
- `cp_kv_cache_interleave_size` Interleave granularity for context-parallel KV-cache layout. It affects how KV slots are distributed across CP ranks and can impose extra runtime constraints on implementations.
- `dcp_comm_backend` Communication/combine backend used for DCP, such as all-gather/reduce-scatter style versus all-to-all style communication. This matters because different DCP execution paths may expect different communication patterns when merging distributed decode attention results.

## Tensor Parallelism (TP)

TP is the most basic multi-GPU mode that directly touches generic attention.

What TP changes:

- local attention heads per rank
- local KV heads per rank
- head divisibility constraints at model validation time

What TP usually does not change:

- `head_size`
- the selector interface shape

Practical consequence:

- most generic backends are written assuming they run on a rank-local shard of heads, not on the global attention head set

The most visible model-level validation is in `vllm/config/model.py`:

- total attention heads must be divisible by `tensor_parallel_size`

## Decode Context Parallelism (DCP)

DCP is the most important distributed feature for backend/kernel requirements.

The useful mental model is:

- TP shards by heads
- DCP shards decode KV by context/time

Important config constraints:

- `tp_size % dcp_size == 0` in `vllm/config/parallel.py`
- for non-MLA GQA/MQA, `tensor_parallel_size > total_num_kv_heads`
- `dcp_size <= tensor_parallel_size // total_num_kv_heads`
- query-heads-per-KV-head must be divisible by `dcp_size`

Important runtime consequence:

- a DCP rank only owns a shard of decode context KV
- softmax normalization is therefore distributed
- the impl must be able to return decode softmax LSE

That requirement is enforced in `vllm/v1/worker/cp_utils.py` through:

- `layer_impl.need_to_return_lse_for_decode`

Impl-side enabling comes from `AttentionImplBase`:

- `can_return_lse_for_decode`
- `need_to_return_lse_for_decode = (dcp_world_size > 1 and can_return_lse_for_decode)`

What these mean:

- `can_return_lse_for_decode` This is the capability flag. It means the implementation knows how to return decode-time softmax log-sum-exp (LSE) in addition to the normal attention output.
- `need_to_return_lse_for_decode = (dcp_world_size > 1 and can_return_lse_for_decode)` This is the runtime-needed flag. It means that for this particular execution, vLLM should actually ask the impl to return LSE. In practice that happens when DCP is active (`dcp_world_size > 1`) and the impl supports it.

So the distinction is:

- `can_return_lse_for_decode` = what the impl is capable of doing
- `need_to_return_lse_for_decode` = what this distributed run requires it to do

So for DCP-capable kernels, returning LSE is not optional bookkeeping. It is part of the correctness contract.

## Prefill Context Parallelism (PCP)

PCP is the prefill-side context-parallel counterpart.

The generic runtime check is also in `vllm/v1/worker/cp_utils.py`:

- impls must advertise `supports_pcp`

So PCP is another example where distributed compatibility is checked at impl runtime, not in the normal backend selector.

What PCP support actually entails:

- PCP is about long-prompt prefill, not decode. The goal is to split a large prefill request across multiple GPUs so time-to-first-token can improve and the per-rank memory burden can drop.
- In the deployment note at `docs/serving/context_parallel_deployment.md`, vLLM describes two PCP-style strategies: partial-query/full-KV and partial-query/partial-KV. In partial-query/full-KV, each rank computes attention for only its local chunk of query tokens, but the full key/value context is gathered so those local queries can still attend to the whole prompt. In partial-query/partial-KV, even the key/value context remains distributed, so ranks must exchange KV chunks progressively, often with a ring-attention-like pattern, instead of materializing the entire prompt KV locally. In both cases, each rank owns only part of the prefill token range, so the impl can no longer assume one rank sees the entire prompt locally in the ordinary single-GPU way.
- Because the prompt is partitioned, a PCP-capable impl must tolerate prefill attention being computed from distributed context ownership. In practice that means the kernel path, metadata path, and any collectives used by the backend must still produce correct prefill outputs when query tokens are sharded across PCP ranks.
- PCP support is represented at impl level, not just backend-class level. `AttentionImplBase` in `vllm/v1/attention/backend.py` carries `supports_pcp`, `pcp_world_size`, `pcp_rank`, `total_cp_world_size`, and `total_cp_rank`, which makes PCP a runtime execution concern for the concrete implementation object.
- PCP also affects KV-cache bookkeeping, not only the math kernel. `SingleTypeKVCacheManager` in `vllm/v1/core/single_type_kv_cache_manager.py` and `UnitaryKVCacheCoordinator` in `vllm/v1/core/kv_cache_coordinator.py` scale effective block-size accounting by context-parallel world size, so one logical scheduler block can correspond to a larger distributed KV region.
- Memory accounting is PCP-aware as well. `FullAttentionSpec.max_memory_usage_bytes(...)` in `vllm/v1/kv_cache_interface.py` divides the effective max-model-length by `dcp_world_size * pcp_world_size`, reflecting that each CP rank stores only part of the full context locally.
- Slot mapping and KV placement are PCP-aware too. `BlockTable.compute_slot_mapping(...)` in `vllm/v1/worker/block_table.py` uses `total_cp_world_size = pcp_world_size * dcp_world_size` and `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`, so PCP changes which rank owns which logical KV slots.

Current practical limitations matter:

- PCP is not "all attention modes automatically work if `supports_pcp` is true". Some KV-cache managers still reject certain patterns.
- `vllm/v1/core/single_type_kv_cache_manager.py` currently asserts that PCP is unsupported for sliding-window attention, chunked local attention, and Mamba.
- `vllm/v1/core/kv_cache_coordinator.py` currently asserts that PCP is unsupported for hybrid attention groups.

So the practical meaning of "PCP support" is:

- the impl advertises `supports_pcp`
- the runtime metadata and communication path can execute sharded prefill correctly
- the KV-cache layout and slot mapping remain valid under PCP
- the specific attention/KV-cache mode in use is one of the currently supported combinations

## CP Interleave Configuration

`ParallelConfig` also carries:

- `cp_kv_cache_interleave_size` Interleave granularity for context-parallel KV-cache placement. It controls how tokens/slots are striped across CP ranks inside a larger virtual KV block.
- legacy `dcp_kv_cache_interleave_size` Older DCP-specific name for the interleave-size setting. Current code treats `cp_kv_cache_interleave_size` as the main knob and uses this legacy field mainly for compatibility/migration.
- `dcp_comm_backend` Which communication pattern DCP should use when ranks exchange or merge distributed decode-attention state, for example all-gather/reduce-scatter style versus all-to-all style.

These values matter because they constrain KV-cache layout and whether certain speculative or MTP paths are valid under context parallelism.

The worker-side CP compatibility check also validates:

- `supports_mtp_with_cp_non_trivial_interleave_size`

This is more specific than ordinary backend selection and is another reason why distributed validation happens after layer/impl construction.

## Pipeline Parallelism (PP)

PP primarily partitions model layers across stages.

For generic attention backend architecture, the important point is:

- PP usually does not change the shape of the attention backend contract
- it changes where layers live, not what an individual backend impl must do

Model support for PP is validated in `vllm/config/model.py`.

## Data Parallelism (DP)

DP mostly replicates model execution across ranks.

For generic attention backend work, DP is usually not a kernel-interface problem:

- each DP rank runs its own attention path
- backend selection and impl contract are mostly unchanged

So DP belongs in the "deployment/runtime topology" bucket more than the "attention backend internals" bucket.

## Expert Parallel / Sequence-Parallel Adjacency

Expert parallel settings live in `ParallelConfig`, but they are mainly MoE communication concerns.

Sequence-parallel related logic appears in compilation config code, especially around size constraints, but it is not a standalone attention backend selector surface in the generic v1 path.

For this research set, it is enough to remember:

- they exist in the broader execution system
- they can constrain compiled execution
- they are not the main place to start when adding a generic attention backend

## Kernel Requirements Under Distributed Attention

For a backend/kernel to be viable in distributed generic attention, the most important requirements are:

- correct local head ownership under TP
- correct KV layout under TP/CP sharding
- decode-LSE return capability for DCP
- PCP support if prefill context parallelism is enabled
- compatibility with configured CP interleave behavior if relevant

For kernels specifically, PCP support usually entails a few extra requirements:

- the prefill path must work when a rank receives only a shard of the query-token range rather than the full prompt
- the kernel or surrounding runtime path must tolerate that the full K/V context may need to be gathered or streamed from other PCP ranks rather than already existing as one local contiguous view
- any sequence-length, position, slot-mapping, or block-table assumptions inside the kernel launch path must remain correct when ownership is expressed in total-CP coordinates rather than single-rank coordinates
- the kernel interface must fit the communication strategy used by the backend, because partial-query/full-KV and partial-query/partial-KV have different data-movement requirements even if the mathematical attention result is the same
- the kernel must still compose correctly with the active KV-cache layout and interleave policy, since PCP changes not only compute partitioning but also how logical token positions map onto physical KV storage

This is why a kernel integration in vLLM is not just:

- can it compute `softmax(QK^T)V`

It is also:

- can it produce the right side-channel data for distributed correctness
- can it work with the KV layout and metadata that the worker constructs

## Where To Read The DCP Path

For a concrete DCP path, read these in order:

- `vllm/config/parallel.py`
- `vllm/config/model.py`
- `vllm/v1/worker/cp_utils.py`
- `vllm/v1/worker/gpu/cp_utils.py`
- `vllm/v1/attention/backends/flash_attn.py`

## Best Files To Read Next

- `vllm/config/parallel.py`
- `vllm/config/model.py`
- `vllm/v1/worker/cp_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
