# Blocks, Prefix Caching, And Eviction

This chapter describes the scheduler's block metadata, prefix-key construction, reference counting, free ordering, and eviction behavior.

The scheduler never stores K/V tensor values itself. It manages identities and ownership of physical block IDs whose bytes live on workers.

## KVCacheBlock

`KVCacheBlock` is defined in `vllm/v1/core/kv_cache_utils.py`.

Fields are:

- `block_id`: physical pool index from zero through `num_gpu_blocks - 1`. `num_gpu_blocks` is the number of KV-cache block slots vLLM decides can fit on the GPU for a worker after memory profiling
- `ref_cnt`: number of active request/group references to the block.
- `_block_hash`: prefix hash plus cache-group ID when a full block is committed to prefix cache.
- `prev_free_block` and `next_free_block`: intrusive linked-list pointers used only by the free queue.
- `is_null`: marks the shared placeholder block that is never prefix-cached or normally freed.

`ref_cnt` is the ownership counter for the scheduler-side block metadata. When a request is actively using a block, or when a cached prefix block is reused by another request, vLLM increments `ref_cnt`. When the request releases that block, vLLM decrements it. A block with `ref_cnt > 0` is not available for normal allocation or eviction. A block with `ref_cnt == 0` can sit on the free queue; if it still has `block_hash`, it is free but still usable as a prefix-cache hit until the allocator evicts/reuses it.

`_block_hash` is the scheduler-side identity for the cached contents of a full block. vLLM hashes the token IDs in a full logical block together with the previous block's hash and any extra keys such as multimodal or LoRA inputs. The cache-group ID is included so two groups with the same token IDs do not collide if their cache semantics differ. `_block_hash` is `None` for newly allocated/uncommitted blocks, partial blocks, evicted blocks, and the null block. Once a full block is committed to prefix cache, the hash lets later requests find and reuse the same physical block instead of recomputing that prefix.

`prev_free_block` and `next_free_block` make each free `KVCacheBlock` act as its own node in the free-block linked list. This is called an intrusive list because the queue links are stored inside the block object itself instead of wrapping blocks in separate queue-node objects. When a block is allocated, removed from the free queue, or returned to the free queue, `FreeKVCacheBlockQueue` updates these two pointers. Active blocks should not be linked in the free list.

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

`FreeKVCacheBlockQueue` in `vllm/v1/core/kv_cache_utils.py` is the ordered list of blocks that are not currently owned by active requests.

The front of the queue is where vLLM takes blocks when it needs space for new KV data. The back of the queue is where ordinary released blocks are placed. So, over time, blocks that have been free for longer tend to move closer to the front and are reused or evicted first.

The queue is implemented as a doubly linked list. Each `KVCacheBlock` stores its own `prev_free_block` and `next_free_block` pointers, so vLLM does not need separate wrapper objects for queue nodes. Fake head and tail sentinel nodes are used internally so insert/remove logic does not need special cases for empty, first, or last elements.

Core operations are:

- `popleft()` and `popleft_n(n)` take one or more blocks from the front for allocation. If a block at the front still has a prefix-cache hash, taking it for new data evicts that cached prefix entry.
- `remove(block)` takes a known block out of the free queue. This is used when prefix caching finds a free cached block by hash and wants to make it active again.
- `append(...)` and `append_n(...)` put normally released blocks at the back, making them less likely to be immediately reused.
- `prepend_n(...)` puts blocks at the front when vLLM wants them to be reclaimed before other free blocks.
- `get_all_free_blocks()` returns the current queue order, mainly for tests and debugging.

At startup, the queue is in block-ID order. After requests allocate, release, touch, and reuse blocks, the order becomes roughly LRU-like: recently used or recently released blocks tend to be farther from the front, while older free blocks tend to be closer to the front.

When a request is freed, vLLM releases its blocks in reverse logical order. For a request with logical blocks `[A, B, C, D]`, where `A` is the earliest prefix block and `D` is the latest tail block, vLLM releases them as `[D, C, B, A]`. Since released blocks are appended to the back of the free queue in that order, `D` ends up closer to the front than `A` among this request's released blocks. If vLLM later needs to reuse one of them, it is more likely to reclaim the less reusable tail block first and preserve earlier prefix blocks longer for prefix-cache hits.

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

`BlockPool` is scheduler-side metadata. It does not contain the actual K/V tensor bytes. The bytes live in worker tensors created from `KVCacheConfig`; the pool only owns `KVCacheBlock` objects that name physical block IDs and track whether those IDs are free, active, cached, or reserved as the null block.

On allocation, `BlockPool.get_new_blocks(...)` takes block objects from the front of `FreeKVCacheBlockQueue`. If a selected free block still has a prefix hash, the pool removes that hash first because the physical slot is about to be overwritten by new K/V data. The returned `KVCacheBlock` objects are then attached to a request's per-group block sequence, and their integer `block_id`s eventually become block-table entries sent to workers.

On release, `BlockPool.free_blocks(...)` decrements each block's `ref_cnt`. Only blocks whose reference count reaches zero are returned to the free queue. If a block still has `_block_hash`, it remains searchable as a prefix-cache entry while free; if allocation later needs that slot, the hash is evicted before reuse.

The prefix hash-to-block map is what lets prefix caching avoid recomputation. When a full block is committed, the pool records its `_block_hash` in the map. Later, `BlockPool.get_cached_block(...)` can find a block with matching prefix contents, remove it from the free queue if necessary, increment its `ref_cnt`, and hand it to the requesting sequence as an already-computed block.

For hybrid cache groups, a single block ID is shared as an allocator coordinate. If group 0 and group 1 both use block ID `17`, they are not sharing one tensor slice of K/V values. Instead, `17` means the corresponding page position in each group's worker-side cache tensors. This is why page-size compatibility matters: the allocator wants one block ID to represent a comparable page unit across all groups using the same pool.

## Null Block

During pool initialization, block zero is removed from the free queue and marked `is_null`.

Managers use this block as a placeholder for token regions that need no real storage, such as old sliding-window pages or padding.

Worker CUDA-graph padding also uses null block ID zero in block-table rows.

The null block's reference count is not managed like normal blocks, so code must avoid freeing or caching it.

## BlockHashToBlockMap

`BlockHashToBlockMap` in `vllm/v1/core/block_pool.py` maps a prefix-cache key to the physical `KVCacheBlock` objects that currently contain that prefix. The key is logically `(block_hash, group_id)`, but vLLM packs it as `BlockHashWithGroupId` in `vllm/v1/core/kv_cache_utils.py` by appending the group ID bytes to the block hash. It is string of block hash and group id.

The `block_hash` part identifies the contents of a full logical prefix block. It is not just a hash of this block's token IDs; it is chained from the previous block hash, so matching block `i` implies the prefix up through block `i - 1` also matched. The `group_id` part disambiguates cache groups. The same token prefix may produce corresponding pages in full-attention, sliding-window, or Mamba-related groups, but those pages do not have identical lifecycle or tensor semantics.

Most keys map to exactly one `KVCacheBlock`, so the map stores that block directly to avoid extra dictionary allocations. If another physical block is later committed with the same key, the entry is upgraded to a small dictionary:

```text
key -> KVCacheBlock(17)

after duplicate cached content appears:
key -> {
  17: KVCacheBlock(17),
  93: KVCacheBlock(93),
}
```

Duplicates are allowed because vLLM does not rewrite already-assigned block tables just to deduplicate identical content. A request may compute and cache a block that another request also computed, and keeping both physical blocks avoids changing block IDs that are already attached to requests.

`get_one_block(key)` returns any one block for that prefix key. If the returned block is free, `BlockPool.touch(...)` can remove it from the free queue and increment `ref_cnt` so it becomes active again. If the returned block is already active, incrementing `ref_cnt` lets another request share it.

`pop(key, block_id)` removes one exact physical block from the map. The `block_id` argument matters when duplicates exist: eviction or reuse of block `17` must remove only block `17` from the prefix-cache map, not another block with the same hash such as block `93`.

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

`init_none_hash(...)` in `vllm/v1/core/kv_cache_utils.py` initializes a parent value used before the first real block hash.

When `PYTHONHASHSEED` is absent, a random process value can be used. CBOR-based modes warn because reproducibility may be lost without a fixed seed in surrounding Python hash-dependent values.

## Extra Hash Keys

Token IDs alone are insufficient when K/V depends on other request inputs.

`generate_block_hash_extra_keys(...)` includes:

- LoRA adapter identity so different adapters cannot share incompatible K/V.
- Multimodal feature identifiers and offsets for placeholder-associated encoder features.
- `cache_salt` on the first block for explicit namespace isolation. A salt is an extra identifier mixed into the first block hash so otherwise-identical token prefixes land in a separate prefix-cache namespace.
- Hashes of prompt embedding slices when embeddings replace token lookup.

Multimodal offsets are relative to the block so otherwise-identical placeholder tokens at different feature positions remain distinct.

## Hash Granularity Conversion

Requests can hash at a finer granularity than a physical group block. This matters for hybrid KV-cache groups because one request has one logical prefix, but different groups may store that prefix using different physical block sizes. For example, a request prefix may be hashed every 8 tokens, while one cache group stores pages of 8 tokens and another cache group stores pages of 32 tokens. Both groups refer to the same request prefix, but the 32-token group needs one hash per 32-token page instead of one hash per 8-token chunk.

`BlockHashListWithBlockSize` in `vllm/v1/core/kv_cache_utils.py` adapts request-level hashes from `hash_block_size` to a larger group block size. It does not recompute token hashes from token IDs. Instead, it lazily concatenates consecutive smaller hash values when the larger group asks for a hash.

Example:

```text
hash_block_size = 8
target group block size = 32
scale_factor = 32 / 8 = 4

request hash entries:
H0 = hash(tokens 0-7)
H1 = hash(tokens 8-15)
H2 = hash(tokens 16-23)
H3 = hash(tokens 24-31)
H4 = hash(tokens 32-39)
H5 = hash(tokens 40-47)
H6 = hash(tokens 48-55)
H7 = hash(tokens 56-63)

group block hashes:
G0 = H0 || H1 || H2 || H3    covers tokens 0-31
G1 = H4 || H5 || H6 || H7    covers tokens 32-63
```

So if `hash_block_size = 8` and a cache group's physical block size is `32`, each group-level cached page advances across four request hash entries.

Only integer scale-up is supported: `target_block_size` must be a multiple of `hash_block_size`. This lets vLLM compute request hashes at the finest common granularity, then provide coarser group-level hashes to groups with larger physical pages. Hybrid groups can therefore participate in one request prefix-cache chain without forcing every group to use the same physical page size.

## Allocating New Blocks

`BlockPool.get_new_blocks(num_blocks)` pops blocks from the queue.

When prefix caching is enabled, each popped block first passes through `BlockPool._maybe_evict_cached_block(...)` in `vllm/v1/core/block_pool.py`.

This method checks whether the free block still has a prefix-cache identity. A free block can still have `block.block_hash` set because vLLM keeps free cached blocks searchable for prefix hits until their physical slot is actually needed. If the block has no hash, `_maybe_evict_cached_block(...)` returns `False` and nothing is removed.

If the block does have a hash, `_maybe_evict_cached_block(...)` removes the exact `(block_hash, block_id)` entry from `cached_block_hash_to_block`. It then clears the block's local hash metadata with `block.reset_hash()`. If KV-cache events are enabled, it emits a `BlockRemoved` event with the external block hash and cache group ID. The important point is that the method evicts the old prefix-cache record before the physical block is reused for different K/V data.

The block's reference count is then raised from zero to one.

Allocation does not inspect prefix hashes because prefix lookup and touching happen before new-block allocation.

## Touching Prefix Hits

`BlockPool.touch(blocks)` acquires existing prefix blocks for a request.

Touch a block increases its reference count by 1, and may remove the block from the free queue. This is used when a block is hit by another request with the same prefix.

If a hit has `ref_cnt == 0`, it is removed from the free queue before incrementing the reference count.

If another request already references it, only the count changes.

Touching also updates residency metrics.

Shared prefix blocks are therefore immutable while referenced. New tokens must use later blocks rather than overwrite a shared full page.

## Committing Full Blocks

`BlockPool.cache_full_blocks(...)` in `vllm/v1/core/block_pool.py` is the point where already-written KV blocks become prefix-cache entries. Before this call, a request may own physical blocks, and workers may have written K/V into those blocks, but the scheduler has not necessarily made those blocks searchable by prefix hash yet.

The caller is the per-group cache manager. In `vllm/v1/core/single_type_kv_cache_manager.py`, it computes:

```text
num_cached_blocks = number of this request's blocks already inserted into prefix cache
num_full_blocks = num_tokens // block_size
```

Only the range:

```text
blocks[num_cached_blocks : num_full_blocks]
```

is new work for `cache_full_blocks(...)`. Earlier blocks are already committed, and later blocks are not complete yet.

For each block in that newly full range, `cache_full_blocks(...)` does the following:

1. Pick the correct hash for this cache group. If the group block size equals `hash_block_size`, it uses `request.block_hashes` directly. If the group block size is larger, it uses `BlockHashListWithBlockSize` to combine smaller request hashes into group-sized hashes.
2. Skip the block if it is the null block or if an optional reachability mask says this block should not participate in prefix-cache lookup for this cache lifecycle.
3. Pack the block hash with `kv_cache_group_id`, producing `BlockHashWithGroupId`.
4. Store that packed hash on the `KVCacheBlock` as `blk.block_hash`.
5. Insert the mapping from packed hash to physical block into `cached_block_hash_to_block`.
6. If KV-cache events are enabled, emit `BlockStored` metadata describing the newly cached hashes and token range.

Only full blocks are inserted into prefix cache. A full block means the block covers exactly `block_size` finalized tokens for that group. For `block_size = 16`, tokens `0..15` can be committed once all 16 are computed, but tokens `16..23` cannot be committed yet because that logical block may later become tokens `16..31`.

Partial tail blocks remain unhashed because their future contents can change as decode appends tokens. If vLLM hashed a partial tail too early, the same physical block would later represent a different token range after more tokens are appended, making the old prefix-cache key stale or incorrect.

Draft tokens are not committed until accepted/finalized. With speculative decoding, some allocated/written blocks may correspond to draft tokens that are later rejected, so prefix-cache commit must follow the accepted token count rather than raw temporary allocation.

Null or unreachable blocks are skipped. This matters for sliding-window, chunked-local, sparse, or Mamba-like lifecycles where some logical positions either have no useful storage or can never be valid prefix-cache hits for that group.

## Why Full Blocks Only

Prefix cache keys describe a fixed token range.

If a partial page were hashed and later extended, either the same hash would refer to changing bytes or every append would require replacing hash identity and lookup state.

Full-page commitment gives an immutable token-to-byte mapping.

## Prefix Lookup

Manager `find_longest_cache_hit(...)` implementations in `vllm/v1/core/single_type_kv_cache_manager.py`, with coordinator wrappers in `vllm/v1/core/kv_cache_coordinator.py`, walk request hashes from the beginning and ask `BlockPool.get_cached_block(...)` for every relevant group.

Lookup stops at the first missing or manager-incompatible page.

The result must be a contiguous prefix. A cached later block cannot be used when an earlier block is absent because attention needs the complete preceding context and chained hashes encode that dependency.

## Recomputing The Last Prompt Token

`KVCacheManager.get_computed_blocks(...)` in `vllm/v1/core/kv_cache_manager.py` caps cache hits at `request.num_tokens - 1`.

This means prefix-cache lookup is not allowed to claim that every prompt token is already computed. If the request has 100 prompt tokens, the maximum cache-hit length is 99 tokens. Even if cached KV exists for all 100 tokens, vLLM intentionally leaves at least one token to run through the model.

Even if the full prompt K/V is cached, vLLM must execute at least one token to produce logits for next-token sampling.

The method passes that cap as `max_cache_hit_length` into `self.coordinator.find_longest_cache_hit(...)`. Because allocation currently requires `num_computed_tokens` to be block-size aligned, this can force recomputation of a full final block rather than exactly one token.

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
