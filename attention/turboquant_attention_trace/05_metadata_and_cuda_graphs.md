# TurboQuant Metadata And CUDA Graphs

`TurboQuantMetadata` and `TurboQuantMetadataBuilder` are defined in `vllm/v1/attention/backends/turboquant_attn.py`.

## Metadata Fields

- `seq_lens: [num_reqs]` contains each request's total valid KV length after including the current step's tokens.
- `slot_mapping: [num_actual_tokens]` maps current tokens to physical cache slots for writes.
- `block_table: [num_reqs, max_num_blocks]` maps each request's logical pages to physical cache blocks for reads.
- `query_start_loc: [num_reqs + 1]` contains cumulative boundaries for the packed current-query tokens.
- `num_actual_tokens` excludes scheduler or CUDA-graph padding tokens.
- `max_query_len` is the largest number of current query tokens owned by one request.
- `max_seq_len` is an upper bound on the longest total KV sequence.
- `is_prefill` is true when `max_query_len > 1`.
- `num_decodes` is the number of decode requests at the beginning of the reordered batch.
- `num_decode_tokens` is the packed token count contributed by those decode requests.
- `query_start_loc_cpu` is a CPU copy used for per-request prefill iteration without a device-to-host synchronization.
- `seq_lens_cpu` is a CPU-resident upper bound used for control flow and sub-batch maxima.

For request `i`, its current query length is:

```text
query_len[i] = query_start_loc[i + 1] - query_start_loc[i]
```

Its previously cached length is:

```text
cached_len[i] = seq_lens[i] - query_len[i]
```

## Builder Input

`TurboQuantMetadataBuilder.build(...)` receives `CommonAttentionMetadata` from `vllm/v1/attention/backend.py`.

The builder does not recreate scheduling information. It selects and renames the common tensors required by TurboQuant, while adding the decode/prefill split used by `forward(...)`.

`common_prefix_len` and `fast_build` are accepted by the standard builder interface but are not used to implement cascade or prefix-specialized TurboQuant attention.

## Decode-First Reordering

The builder constructor calls `_init_reorder_batch_threshold(1, supports_spec_as_decode=False)`.

A query length of one is treated as decode for reordering purposes. The model runner arranges those requests first, and `split_decodes_and_prefills(...)` from `vllm/v1/attention/backends/utils.py` returns the request and token boundary.

This allows `forward(...)` to slice one contiguous decode prefix and one contiguous prefill suffix instead of gathering arbitrary request positions.

Speculative requests with more than one query token are not treated as ordinary decode by this builder because `supports_spec_as_decode` is false.

## Pure And Mixed Batches

`is_prefill` is a batch-level indicator. A mixed batch has `is_prefill = True` because at least one request has multiple query tokens, while `num_decodes > 0` identifies the decode prefix.

The implementation therefore dispatches as follows:

- `is_prefill == False`: all requests use decode.
- `is_prefill == True` and `num_decodes == 0`: all requests use prefill.
- `is_prefill == True` and `num_decodes > 0`: split the packed tensors and metadata into decode and prefill sub-batches.

## Mixed-Batch Metadata Slicing

The decode sub-metadata slices request tensors through `num_decodes` and token tensors through `num_decode_tokens`.

`query_start_loc` initially contains offsets into the complete packed token tensor, whose layout is `[decode tokens | prefill tokens]`. Once `forward(...)` removes the decode prefix and passes only the prefill suffix onward, the first prefill token must become position zero. The builder therefore subtracts `num_decode_tokens` from every prefill offset.

For example, consider `[D0 D1 | P0 P1 P2 | P3 P4]`. The two prefill requests start at positions 2 and 5 in the complete tensor, so their original cumulative offsets are `[2, 5, 7]`. There are two decode tokens, so subtracting `num_decode_tokens = 2` produces `[0, 3, 5]`. These adjusted offsets correctly describe the prefill-only tensor `[P0 P1 P2 | P3 P4]`.

The prefill maximum sequence length is computed from the CPU-resident prefill suffix when available. This prevents long decode contexts from inflating the prefill maximum and disabling the first-chunk FlashAttention fast-path test `max_query_len == max_seq_len`.

## CUDA Graph Support

The builder declares `AttentionCGSupport.UNIFORM_BATCH`.

This enum means CUDA graph execution is supported when requests in the captured attention batch have a uniform query-length pattern, including speculative decode with a fixed number of tokens. It does not promise arbitrary mixed prefill/decode capture.

`build_for_cudagraph_capture(...)` calls the normal builder and fills `seq_lens` with one. This keeps capture-time decode work small; replay updates the input buffers with real sequence lengths.

The decode launcher uses a fixed `NUM_KV_SPLITS = tq_max_kv_splits_for_cuda_graph`, so stage-one grid dimension 2 and the intermediate tensor shape do not vary with runtime sequence length.

The relevant config is `AttentionConfig.tq_max_kv_splits_for_cuda_graph` in `vllm/config/attention.py`, with a default of 32.

## Other Builder Contract Methods

`get_cudagraph_support(...)` is inherited and returns the class-level `UNIFORM_BATCH` value.

`build_for_drafting(...)` is inherited and calls `build(common_prefix_len=0, fast_build=True)`. TurboQuant's `build(...)` currently accepts `fast_build` but constructs the same metadata fields either way.

`supports_update_block_table` remains false, so `update_block_table(...)` is not implemented for TurboQuant metadata. The model runner must build the appropriate metadata instead of asking this builder to replace block tables in place.

`use_cascade_attention(...)` is inherited and returns false. TurboQuant does not construct cascade-attention metadata from a common batch prefix.

## Metadata Builder Lifetime

`GPUModelRunner.initialize_attn_backend(...)` groups compatible attention layers, and `initialize_metadata_builders(...)` constructs one backend-specific builder per attention group.

Layers can share a builder when they have the same backend class, compatible KV-cache spec, and matching local query-head grouping requirements. Runtime forward context then provides the resulting metadata to each layer in that group.
