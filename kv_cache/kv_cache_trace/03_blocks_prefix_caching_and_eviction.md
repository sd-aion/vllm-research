# Blocks, Prefix Caching, And Eviction

This chapter describes the scheduler's block metadata, prefix-key construction, reference counting, free ordering, and eviction behavior.

The scheduler never stores K/V tensor values itself. It manages identities and ownership of physical block IDs whose bytes live on workers.

## KVCacheBlock

`KVCacheBlock` is defined in `vllm/v1/core/kv_cache_utils.py`.

Fields are:

- `block_id`: physical pool index from zero through `num_gpu_blocks - 1`.
- `ref_cnt`: number of active request/group references to the block.
- `_block_hash`: prefix hash plus cache-group ID when a full block is committed to prefix cache.
- `prev_free_block` and `next_free_block`: intrusive linked-list pointers used only by the free queue.
- `is_null`: marks the shared placeholder block that is never prefix-cached or normally freed.

A block can be both free and prefix-cached.

That state means `ref_cnt == 0`, the block is on the free queue, and `block_hash` is still present. Its bytes are reusable as a prefix hit until allocation reaches it and evicts the hash.

## Block State Model

A useful state model is:

```text
unused free:       ref_cnt=0, hash=None, on free queue
active unhashed:   ref_cnt>0, hash=None, not on free queue
active cached:     ref_cnt>0, hash=set, not on free queue
free cached:       ref_cnt=0, hash=set, on free queue and eligible as prefix hit
null placeholder:  is_null=True, special lifetime, hash=None
```

Physical bytes are not cleared during ordinary transitions. Correctness comes from block-table ownership, sequence lengths, slot writes, and hash invalidation before reuse.

## FreeKVCacheBlockQueue

`FreeKVCacheBlockQueue` in `vllm/v1/core/kv_cache_utils.py` is an intrusive doubly linked list.

It avoids allocating queue-node objects because links live directly on `KVCacheBlock`.

Fake head and tail sentinels reduce branch handling.

Core operations are:

- `popleft()` and `popleft_n(n)` allocate from the current eviction front.
- `remove(block)` removes a prefix-hit block from the middle in O(1).
- `append(...)` and `append_n(...)` place released blocks at the back.
- `prepend_n(...)` places blocks at the front when they should be reclaimed first.
- `get_all_free_blocks()` exposes order for tests.

The queue starts in block-ID order.

After runtime use, ordering approximates LRU. Request blocks are freed in reverse logical order so deeper tail blocks become earlier eviction candidates when access times are otherwise equal.

## BlockPool

`BlockPool` is defined in `vllm/v1/core/block_pool.py`.

It owns:

- The fixed list of all `KVCacheBlock` objects.
- The free queue.
- The prefix hash-to-block map.
- The shared null block.
- Optional cache events.
- Optional residency metrics.

One `BlockPool` can serve multiple cache groups because one block ID represents corresponding pages across groups.

## Null Block

During pool initialization, block zero is removed from the free queue and marked `is_null`.

Managers use this block as a placeholder for token regions that need no real storage, such as old sliding-window pages or padding.

Worker CUDA-graph padding also uses null block ID zero in block-table rows.

The null block's reference count is not managed like normal blocks, so code must avoid freeing or caching it.

## BlockHashToBlockMap

`BlockHashToBlockMap` maps `(block_hash, group_id)` keys to one or more blocks.

Multiple physical blocks can carry the same prefix hash because concurrent computation or race-like reuse can create duplicate cached content.

Lookup returns one available block, while removal includes block ID so eviction removes the exact physical instance.

Group ID is part of the key because block ID pages in different cache groups have different byte contents even when they correspond to the same token prefix.

## Chained Prefix Hashes

Request block hashes are generated in `vllm/v1/core/kv_cache_utils.py` and stored on the `Request`.

Conceptually, hash block `i` depends on:

```text
hash_i = H(parent_hash, token_ids_i, extra_keys_i)
parent_hash = hash_(i-1)
```

The chain ensures a token block only matches when every preceding block also matches.

Two identical token chunks at different prefix positions do not collide semantically because their parent hashes differ.

## Hash Algorithms

`prefix_caching_hash_algo` selects:

- `sha256`: Pickle serialization followed by SHA-256.
- `sha256_cbor`: canonical CBOR-style serialization followed by SHA-256.
- `xxhash`: Pickle serialization followed by 128-bit xxHash.
- `xxhash_cbor`: canonical serialization followed by xxHash.

SHA-256 prioritizes collision resistance.

xxHash reduces hash cost but is not cryptographically collision-resistant, which matters for multi-tenant isolation.

Canonical serialization supports reproducible hashes across compatible language/process implementations.

## NONE_HASH

`init_none_hash(...)` initializes a parent value used before the first real block hash.

When `PYTHONHASHSEED` is absent, a random process value can be used. CBOR-based modes warn because reproducibility may be lost without a fixed seed in surrounding Python hash-dependent values.

## Extra Hash Keys

Token IDs alone are insufficient when K/V depends on other request inputs.

`generate_block_hash_extra_keys(...)` includes:

- LoRA adapter identity so different adapters cannot share incompatible K/V.
- Multimodal feature identifiers and offsets for placeholder-associated encoder features.
- `cache_salt` on the first block for explicit namespace isolation.
- Hashes of prompt embedding slices when embeddings replace token lookup.

Multimodal offsets are relative to the block so otherwise-identical placeholder tokens at different feature positions remain distinct.

## Hash Granularity Conversion

Requests can hash at a finer granularity than a physical group block.

`BlockHashListWithBlockSize` combines consecutive fine-grained hashes to provide hashes at a larger group's block size.

If `hash_block_size = 8` and group block size is 32, each group-level cached page advances across four request hash entries.

This lets hybrid groups participate in one request prefix without forcing identical physical page sizes.

## Allocating New Blocks

`BlockPool.get_new_blocks(num_blocks)` pops blocks from the queue.

When prefix caching is enabled, each popped block first passes through `_maybe_evict_cached_block(...)`.

If the block had a hash, eviction removes it from the hash map, clears its hash metadata, records metrics, and optionally emits `BlockRemoved`.

The block's reference count is then raised from zero to one.

Allocation does not inspect prefix hashes because prefix lookup and touching happen before new-block allocation.

## Touching Prefix Hits

`BlockPool.touch(blocks)` acquires existing prefix blocks for a request.

If a hit has `ref_cnt == 0`, it is removed from the free queue before incrementing the reference count.

If another request already references it, only the count changes.

Touching also updates residency metrics.

Shared prefix blocks are therefore immutable while referenced. New tokens must use later blocks rather than overwrite a shared full page.

## Committing Full Blocks

`BlockPool.cache_full_blocks(...)` assigns hashes to newly full, finalized request blocks.

Inputs identify:

- The request and its precomputed hash chain.
- All request blocks in one group.
- Number already committed.
- Number now full and committable.
- Group block size and group ID.
- Optional reachability mask for windows or sparse lifecycles.

Only full blocks are inserted into prefix cache.

Partial tail blocks remain unhashed because their future contents can change as decode appends tokens.

Draft tokens are not committed until accepted/finalized.

Null or unreachable blocks are skipped.

## Why Full Blocks Only

Prefix cache keys describe a fixed token range.

If a partial page were hashed and later extended, either the same hash would refer to changing bytes or every append would require replacing hash identity and lookup state.

Full-page commitment gives an immutable token-to-byte mapping.

## Prefix Lookup

Manager `find_longest_cache_hit(...)` implementations walk request hashes from the beginning and ask `BlockPool.get_cached_block(...)` for every relevant group.

Lookup stops at the first missing or manager-incompatible page.

The result must be a contiguous prefix. A cached later block cannot be used when an earlier block is absent because attention needs the complete preceding context and chained hashes encode that dependency.

## Recomputing The Last Prompt Token

`KVCacheManager.get_computed_blocks(...)` caps cache hits at `request.num_tokens - 1`.

Even if the full prompt K/V is cached, vLLM must execute at least one token to produce logits for next-token sampling.

Because allocation currently aligns computed-token counts to block boundaries, this can force recomputation of a full final block rather than exactly one token.

## Releasing Blocks

`BlockPool.free_blocks(...)` decrements each block's reference count.

Blocks reaching zero are inserted into the free queue but retain their hash when prefix caching is enabled.

`prepend=True` makes them immediate reuse candidates; normal request completion generally appends in an order chosen by the manager.

Blocks with remaining references stay active and cannot be allocated.

## Eviction Versus Freeing

Freeing changes ownership by reducing `ref_cnt` and potentially placing the block on the free queue.

Eviction removes prefix lookup metadata by clearing the hash.

An active block can be evicted from prefix cache without being physically freed; its current request can still use its block-table entry.

`BlockPool.evict_blocks(block_ids)` is used when a connector reports that loaded contents are invalid or polluted.

## Explicit Prefix Reset

`BlockPool.reset_prefix_cache()` succeeds only when no normal blocks are actively referenced; the null block is the one expected used block.

It replaces the hash map, clears every block hash, resets residency metrics, and emits `AllBlocksCleared` when events are enabled.

This is used after weight changes, such as RLHF updates, because cached K/V produced by old weights is invalid.

## Cache Events

Event classes live in `vllm/distributed/kv_events.py`.

`BlockStored` reports hashes, parent hash, token IDs, block size, LoRA identity, medium, extra keys, and group index.

`BlockRemoved` reports removed hashes, storage medium, and group index.

`AllBlocksCleared` marks complete invalidation.

These events support external cache coordination and observability without exposing raw K/V bytes.

## Residency Metrics

`KVCacheMetricsCollector` in `vllm/v1/core/kv_cache_metrics.py` tracks allocation, access, eviction, and residency state.

Scheduler `PrefixCacheStats` records queried tokens, hits, misses, and preemption-related behavior.

Connector prefix-cache stats are tracked separately because an external hit does not imply a local GPU prefix hit.

## Worked Sharing Example

Assume block size four and request A tokens `[10, 11, 12, 13, 20, 21]`.

A receives blocks 7 and 9. After computing the first four tokens, block 7 becomes full and receives hash `h0`; block 9 remains unhashed because it holds only two valid tokens.

When A finishes, both references drop to zero. Block 7 remains in the free queue with `h0`; block 9 is free and unhashed.

Request B begins with `[10, 11, 12, 13, 99]`. Prefix lookup finds block 7, removes it from the free queue, and raises its reference count.

B allocates another block for token 99. It never overwrites block 7, so A's cached prefix remains immutable and can be shared by additional requests.

If later allocation pressure reaches free cached block 7 after all references return to zero, `_maybe_evict_cached_block(...)` removes `h0` before the physical block is reused.
