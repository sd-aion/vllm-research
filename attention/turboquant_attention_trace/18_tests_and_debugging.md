# TurboQuant Tests And Debugging

The primary test file is `tests/quantization/test_turboquant.py`.

## Existing CPU Coverage

`TestTurboQuantConfig` checks:

- Parsing every named preset.
- Rejection of an unknown preset.
- Key and value bit modes.
- Centroid counts and norm-correction flags.
- Exact packed sizes for `D = 128`.
- Slot-size and alignment invariants.
- Configuration arithmetic for head dimensions 64, 96, 128, and 256.
- Dense boundary-layer skip behavior.

`TestHybridAttentionIndices` checks extraction of full-attention layer indices from the model-config conventions used by hybrid architectures.

`TestCentroids` and `TestLloydMax` check centroid shapes, ordering, symmetry, determinism, midpoint relationships, and numerical agreement with a reference.

These tests validate configuration and quantizer construction but do not prove that every advertised head dimension is executable by the Hadamard serving path.

## Existing Device Coverage

`TestHadamardRotation` checks orthonormality and symmetry for dimensions 64, 128, and 256.

`TestStoreDecodeRoundTrip.test_single_token_roundtrip` executes store followed by decode for:

- `turboquant_k8v4`.
- `turboquant_4bit_nc`.

It uses one logical token, four query heads, four KV heads, one cache page, and `D = 128`. Because softmax over one key is exactly one, output should approximate the quantized stored value, and the test checks per-head cosine similarity.

## Current Coverage Gaps

The existing round-trip test does not cover:

- `turboquant_k3v4_nc` or `turboquant_3bit_nc` device execution.
- Multiple logical tokens or cache pages.
- Nontrivial block tables and slot mappings.
- GQA or MQA head mapping.
- bf16 Q/K/V.
- First-chunk prefill delegation.
- Short continuation synthetic decode.
- Large continuation full dequantization.
- Mixed decode/prefill batches.
- CUDA graph capture and replay.
- Workspace fallback versus workspace-manager buffers.
- Tensor, pipeline, or context parallel execution.
- Boundary-layer mixed cache specs.

## Running The Test File

From `/home/ubuntu/vllm`:

```bash
.venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
```

Device tests skip when the test platform does not expose a supported GPGPU.

## Debugging Cache Layout

For a chosen preset and head dimension, first compute:

```text
MSE_BYTES = ceil(D * key_mse_bits / 8)
KPS = key_packed_size
VAL_DATA_BYTES = ceil(D * value_bits / 8)
slot_size = KPS + VAL_DATA_BYTES + 4
```

Then confirm:

- `kv_cache.dtype == torch.uint8`.
- `kv_cache.shape[-1] == slot_size_aligned`.
- `kv_cache.shape[2] == local num_kv_heads`.
- Store and decode use the same `KPS`, bit widths, FP8 format, and norm-correction flag.

A disagreement in any one of these constants shifts subsequent fields and makes both key scores and values invalid.

## Debugging Slot Mapping

Choose one token and calculate `block = slot // block_size` and `offset = slot % block_size` manually.

Confirm that the request's block table maps the corresponding logical page to that same physical block when decode later reads the token.

If writes appear missing, check for negative slot mappings and verify `num_actual_tokens` slicing before the store launcher.

## Debugging MSE Keys

Check these stages independently:

1. `PiT @ Pi` is approximately identity and the matrix shape is exactly `[D, D]`.
2. Normalized rotated keys have row norms near one.
3. Midpoints are sorted and have `2^bits - 1` entries.
4. Packed indices stay within `[0, 2^bits - 1]` after unpacking.
5. Stored fp16 norms match the original key norms within fp16 error.
6. Query rotation uses the same matrix orientation as key rotation.

For norm-corrected presets, compare score reconstruction both before and after centroid-vector normalization.

## Debugging Values

For one token-head vector, inspect `v_min`, `v_max`, and expected scale.

Unpack a few coordinates by hand, then verify `index * scale + minimum` against the source values. Boundary-crossing 3-bit coordinates are especially useful because they test the two-byte extraction path.

## Debugging Split-KV Decode

Start with `NUM_KV_SPLITS = 1` to isolate packing and score correctness, then increase the split count to test LSE reduction.

Compare stage-one split outputs and LSE against a dense reference. Verify that stage two matches a direct softmax over the union of all splits rather than an arithmetic average of split outputs.

For a short sequence with many fixed splits, confirm empty splits return in stage one and are skipped in stage two.

## Debugging Continuation Prefill

For the synthetic-decode path, verify that a chunk with cached length `C` receives sequence lengths `[C+1, C+2, ..., C+q_len]` and repeated views of the same block-table row.

For the full-dequant path, compare reconstructed cached K/V against a standalone unpack reference before concatenating current raw K/V. For MSE keys, remember that `_tq_full_dequant_kv` output must still be inverse-rotated.

## Useful Source Search

```bash
rg -n "TurboQuant|turboquant_|_tq_" vllm tests/quantization/test_turboquant.py
```

The most useful execution boundary is the pair `unified_kv_cache_update` and `unified_attention_with_output` in `vllm/model_executor/layers/attention/attention.py`.
