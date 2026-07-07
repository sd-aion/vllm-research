# vLLM Research Notes

This repository contains implementation-oriented research notes for vLLM. It is intended to grow across vLLM subsystems as new topics are investigated.

The current material focuses on attention backends and KV-cache management, including launch configuration, backend selection, metadata construction, cache allocation, kernel execution, distributed attention, cache transfer, and offloading. Future research can add independent areas such as scheduling, model loading, distributed serving, compilation, quantization, speculative decoding, multimodal execution, or other vLLM internals.

The vLLM source tree used by these notes is `/home/ubuntu/vllm`. Source paths inside the documents are relative to that directory unless stated otherwise.

## Current Research Areas

| Area | Start here | Coverage |
| --- | --- | --- |
| Generic attention architecture | [attention/README.md](attention/README.md) | Backend selection, registration, contracts, layer construction, runtime execution, metadata, distributed attention, sliding-window attention, and multimodal prefix attention. |
| FlashAttention backend | [attention/flash_attention_trace/README.md](attention/flash_attention_trace/README.md) | FlashAttention versions, backend capabilities, metadata, paged KV cache, prefill, decode, external kernels, distributed execution, and cascade attention. |
| Triton attention backend | [attention/triton_attention_trace/README.md](attention/triton_attention_trace/README.md) | vLLM-owned Triton cache-update, unified attention, segmented decode, reduction, encoder attention, helper, and adjacent paged-decode kernels. |
| TurboQuant backend | [attention/turboquant_attention_trace/README.md](attention/turboquant_attention_trace/README.md) | Compressed KV-cache algorithm, cache layout, store kernels, prefill, continuation prefill, split-KV decode, full dequantization, and limitations. |
| KV-cache management | [kv_cache/kv_cache_trace/README.md](kv_cache/kv_cache_trace/README.md) | Configuration, cache specs, profiling, physical allocation, block management, prefix caching, block tables, slot mapping, KV connectors, CPU offloading, and backend integration. |
| TurboQuant KV benchmark | [turboquant_kv_benchmark/README.md](turboquant_kv_benchmark/README.md) | Pure-PyTorch correctness benchmark comparing vLLM-style TurboQuant KV cache reconstruction against paper-style MSE and MSE+QJL TurboQuant variants. |

## Current Scope Boundaries

The documents currently present focus on vLLM v1 generic attention, the FlashAttention, Triton, and TurboQuant backends, and KV-cache management. MLA internals, Mamba state management, ROCm-only backends, legacy v0 execution, and other vLLM subsystems are not yet comprehensively mapped; this is a description of current coverage rather than a permanent exclusion.

These documents describe the local source checkout and may become stale as vLLM evolves. When implementing a backend, verify every claimed capability and call signature against the current source.
