# Barq Inference - Phased Implementation Plan

## Overview

This plan addresses the gaps identified in `GAP_ANALYSIS.md` through focused, atomic implementation phases. Each phase builds on previous work and can be tested independently.

**Strategy:**
1. Start with high-impact quantization improvements
2. Expand model architecture support
3. Add GPU backend support
4. Implement advanced features
5. Polish with optimizations and extras

---

## Phase 17: Quantization Types Expansion (P0)

**Goal:** Implement critical missing quantization types for memory efficiency and model compatibility.

### Phase 17.1: Q2_K and Q3_K Quantization
- [x] Add Q2_K quantization struct in `quant/src/q2_k.rs`
- [x] Add Q3_K quantization struct in `quant/src/q3_k.rs`
- [x] Implement quantize/dequantize for both
- [x] Add SIMD optimizations (AVX2)
- [x] Add unit tests
- [x] Wire into `barq_core/src/gguf.rs` for loading

### Phase 17.2: Q5_K Quantization
- [x] Add Q5_K quantization struct in `quant/src/q5_k.rs`
- [x] Implement block quantization with scales
- [x] Add SIMD optimizations
- [x] Add unit tests
- [x] Integrate with GGUF loader

### Phase 17.3: IK Quantization Integration
- [x] Complete `quant/src/ik_quant.rs` implementation
- [x] Implement IQ4_KS, IQ3_KS, IQ2_KS quantization
- [x] Implement Q4_K_R4 row-interleaved variant
- [x] Add `quantize_model_ik()` function
- [x] Add `repack_model_cpu()` for conversion
- [x] Add comprehensive tests

### Phase 17.4: KV Cache Quantization
- [x] Add Q8_KV quantization in `quant/src/q8_kv.rs`
- [x] Integrate with `models/src/kv_cache.rs`
- [x] Add runtime quantization flag
- [x] Implement defragmentation for quantized cache
- [x] Benchmark memory savings

---

## Phase 18: Model Architecture Expansion (P1)

**Goal:** Support the most commonly used model architectures.

### Phase 18.1: Qwen Architecture Family
- [x] Create `models/src/qwen.rs` for Qwen base
- [x] Create `models/src/qwen2.rs` for Qwen2
- [x] Create `models/src/qwen3.rs` for Qwen3
- [x] Implement RoPE with NTK-aware scaling
- [x] Add GQA support for Qwen models
- [x] Add test GGUF files

### Phase 18.2: DeepSeek Architecture
- [x] Create `models/src/deepseek.rs`
- [x] Implement Multi-head Latent Attention (MLA)
- [x] Add MoE routing for DeepSeek-MoE
- [x] Implement FFN with SwiGLU
- [x] Add test cases

### Phase 18.3: Mistral/Mixtral Improvements
- [x] Complete `models/src/mistral.rs` implementation
- [x] Complete `models/src/mixtral.rs` with full MoE routing
- [x] Add sliding window attention
- [x] Implement expert load balancing
- [x] Add benchmarking

### Phase 18.4: Architecture Registry
- [x] Create `models/src/arch_registry.rs`
- [x] Map GGUF architecture names to implementations
- [x] Add architecture detection from metadata
- [x] Create unified `LlmArch` trait
- [x] Document all supported architectures

---

## Phase 19: Grammar System (P1)

**Goal:** Enable structured output generation.

### Phase 19.1: Grammar Parser
- [x] Create `barq_core/src/grammar/` module
- [x] Implement GBNF (GGML BNF) parser
- [x] Add grammar AST representation
- [x] Implement grammar compilation
- [x] Add unit tests

### Phase 19.2: Grammar-Guided Sampling
- [x] Create `sampling/src/grammar_sampler.rs`
- [x] Implement token masking from grammar
- [x] Add grammar state tracking
- [x] Integrate with existing sampler chain
- [x] Test with JSON grammar

### Phase 19.3: JSON Mode
- [x] Create `sampling/src/json_mode/mod.rs`
- [x] Implement JSON schema to grammar conversion
- [x] Add `--json` CLI flag
- [x] Add validation of output
- [x] Document usage

---

## Phase 20: CPU Optimization (P1)

**Goal:** Maximize CPU inference performance.

### Phase 20.1: AVX-512 Kernels
- [x] Add AVX-512 detection in `barq_core/src/platform.rs`
- [x] Implement AVX-512 matrix multiplication
- [x] Implement AVX-512 quantization kernels
- [x] Add runtime dispatch based on CPU features
- [x] Benchmark vs AVX2 baseline

### Phase 20.2: ARM NEON Improvements
- [x] Audit existing NEON code
- [x] Add missing NEON intrinsics
- [x] Implement NEON GEMM kernels
- [x] Test on Apple Silicon
- [x] Document performance

### Phase 20.3: GEMM Optimizations
- [x] Implement blocked GEMM in `barq_core/src/gemm.rs`
- [x] Add cache-aware tiling
- [x] Implement packing for cache efficiency
- [x] Add threading with rayon
- [x] Benchmark different sizes

### Phase 20.4: Prompt Processing Optimization
- [x] Optimize prompt encoding path
- [x] Implement parallel prompt processing
- [x] Add batched prompt caching
- [x] Measure TTFT improvements
- [x] Document results

---

## Phase 21: CUDA Backend (P0)

**Goal:** Enable NVIDIA GPU inference.

**Implementation note:** The CUDA work landed in the existing backend CUDA modules rather than the placeholder submodule layout below.

### Phase 21.1: CUDA Context and Memory
- [x] Create `backend/src/cuda/context.rs`
- [x] Implement CUDA device detection
- [x] Implement GPU memory allocation
- [x] Add CUDA error handling
- [x] Test device initialization

### Phase 21.2: CUDA GEMM Kernels
- [x] Create `backend/src/cuda/kernels.rs`
- [x] Implement cuBLAS integration
- [x] Add quantization CUDA kernels
- [x] Implement matrix operations
- [x] Benchmark vs CPU

### Phase 21.3: CUDA Attention
- [x] Implement Flash Attention CUDA
- [x] Add KV cache GPU management
- [x] Implement RoPE on GPU
- [x] Add softmax kernel
- [x] Test with model

### Phase 21.4: CUDA Pipeline Integration
- [x] Create `backend/src/cuda/pipeline.rs`
- [x] Implement layer-by-layer GPU execution
- [x] Add CPU fallback path
- [x] Implement tensor transfers
- [x] End-to-end testing

---

## Phase 22: Metal Backend (P1)

**Goal:** Enable Apple Silicon GPU inference.

**Implementation note:** The Metal backend work is implemented in the existing backend Metal modules rather than the placeholder submodule layout below.

### Phase 22.1: Metal Device Setup
- [x] Create `backend/src/metal/device.rs`
- [x] Implement Metal device detection
- [x] Create Metal command queue
- [x] Add buffer management
- [x] Test on macOS

### Phase 22.2: Metal Shaders
- [x] Create Metal shader files (`.metal`)
- [x] Implement GEMM shader
- [x] Implement attention shader
- [x] Implement quantization shaders
- [x] Compile shaders at build time

### Phase 22.3: Metal Integration
- [x] Create `backend/src/metal/pipeline.rs`
- [x] Implement tensor operations
- [x] Add KV cache Metal buffer
- [x] Integrate with model inference
- [x] Benchmark on Apple Silicon

---

## Phase 23: FlashMLA and MoE Optimizations (P2)

**Goal:** Optimize DeepSeek and MoE model inference.

### Phase 23.1: FlashMLA Implementation
- [x] Create `advanced/src/flash_mla.rs`
- [x] Implement MLA-2 (K-cache only)
- [x] Implement MLA-3 (optimized)
- [x] Add `-mla` CLI flag
- [x] Document usage

### Phase 23.2: MoE Fused Operations
- [x] Create `models/src/moe_fused.rs`
- [x] Implement fused FFN up/gate
- [x] Implement expert batch processing
- [x] Add `-fmoe` CLI flag
- [x] Benchmark MoE speedup

### Phase 23.3: Smart Expert Reduction
- [x] Implement SER for DeepSeek
- [x] Add `-ser` CLI flag
- [x] Add expert routing optimization
- [x] Test accuracy vs speed
- [x] Document trade-offs

---

## Phase 24: Chat Templates and Server (P2)

**Goal:** Production-ready serving capabilities.

### Phase 24.1: Jinja Chat Templates
- [x] Create `vocab/src/chat_template.rs`
- [x] Implement Jinja2 template parser
- [x] Add common templates (LLaMA, Mistral, Qwen, etc.)
- [x] Implement template application
- [x] Test with various models

### Phase 24.2: HTTP Server
- [x] Create `cli/src/server.rs`
- [x] Implement `/v1/chat/completions` endpoint
- [x] Implement `/v1/completions` endpoint
- [x] Add streaming support (SSE)
- [x] Add CORS support

### Phase 24.3: OpenAI API Compatibility
- [x] Implement `/v1/models` endpoint
- [x] Implement `/v1/responses` (ik_llama.cpp style)
- [x] Add token counting
- [x] Add rate limiting
- [x] Document API compatibility

---

## Phase 25: Testing Infrastructure (P1)

**Goal:** Ensure correctness and prevent regressions.

### Phase 25.1: Unit Test Expansion
- [x] Add tests for each quantization type
- [x] Add tests for each architecture
- [x] Add tests for sampling algorithms
- [x] Add tests for attention mechanisms
- [x] Achieve >80% coverage

### Phase 25.2: Integration Tests
- [x] Create `cli/tests/integration/`
- [x] Add model loading tests
- [x] Add end-to-end inference tests
- [x] Add memory leak tests
- [x] Add CI workflow

### Phase 25.3: Benchmark Suite
- [x] Create `cli/benches/` with criterion benchmarks
- [x] Add token generation benchmarks
- [x] Add prompt processing benchmarks
- [x] Add memory benchmarks
- [x] Track performance over time

---

## Phase 26: Multimodal Foundation (P3)

**Goal:** Enable vision-language model support.

### Phase 26.1: Vision Encoder Interface
- [x] Create `models/src/vision/mod.rs`
- [x] Define vision encoder trait
- [x] Add CLIP-like encoder support
- [x] Add image preprocessing
- [x] Test embedding extraction

### Phase 26.2: Vision-Language Models
- [x] Create `models/src/qwen2vl.rs`
- [x] Create `models/src/llava.rs`
- [x] Implement cross-attention
- [x] Add image token handling
- [x] Test with multimodal GGUF

---

## Phase 27: Documentation and Polish (P2)

**Goal:** Production-ready documentation.

### Phase 27.1: API Documentation
- [ ] Document all public APIs
- [ ] Add rustdoc examples
- [ ] Add architecture diagrams
- [ ] Document quantization choices
- [ ] Document backend options

### Phase 27.2: User Documentation
- [ ] Update README with all features
- [ ] Create usage examples
- [ ] Document CLI flags
- [ ] Create migration guide from llama.cpp
- [ ] Document performance tuning

### Phase 27.3: Developer Documentation
- [ ] Create CONTRIBUTING.md
- [ ] Document architecture decisions
- [ ] Create module diagrams
- [ ] Document testing approach
- [ ] Document release process

---

## Implementation Priority Order

```
P0 (Critical):
├── Phase 17: Quantization Types
├── Phase 21: CUDA Backend
└── Phase 18: Model Architectures (core ones)

P1 (High):
├── Phase 19: Grammar System
├── Phase 20: CPU Optimization
├── Phase 22: Metal Backend
└── Phase 25: Testing Infrastructure

P2 (Medium):
├── Phase 23: FlashMLA and MoE
├── Phase 24: Chat Templates/Server
└── Phase 27: Documentation

P3 (Lower):
└── Phase 26: Multimodal Foundation
```

---

## Per-Commit Guidelines

Each phase consists of **atomic commits** following this pattern:

```
Phase X.Y: <Component Name>

- Add <specific structure/function>
- Implement <specific algorithm>
- Add unit tests for <feature>
- Wire into <module>
```

**Commit Format:**
```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`

---

## Branch Strategy

```
main
├── phase-17.1-q2k-q3k
├── phase-17.2-q5k
├── phase-17.3-ik-quant
├── ...
└── phase-27.3-dev-docs
```

Each phase.X.Y gets its own branch, commits are atomic, and branches are pushed after completion.

---

## Progress Tracking

| Phase | Status | Branch | Key Deliverable |
|-------|--------|--------|-----------------|
| 17.1 | Done | main | Q2_K, Q3_K quantization |
| 17.2 | Done | main | Q5_K quantization |
| 17.3 | Done | main | IK quantization |
| 17.4 | Done | main | KV cache quantization |
| 18.1 | Done | main | Qwen architecture |
| 18.2 | Done | main | DeepSeek architecture |
| 18.3 | Done | main | Mistral improvements |
| 18.4 | Done | main | Architecture registry |
| 19.1 | Done | phase_19_grammar_json | Grammar parser |
| 19.2 | Done | phase_19_grammar_json | Grammar sampling |
| 19.3 | Done | phase_19_grammar_json | JSON mode |
| 20.1 | Done | phase_20_cpu_optimizations | AVX-512 kernels |
| 20.2 | Done | phase_20_cpu_optimizations | NEON improvements |
| 20.3 | Done | phase_20_cpu_optimizations | GEMM optimizations |
| 20.4 | Done | phase_20_cpu_optimizations | Prompt optimization |
| 21.1 | Done | phase_21_cuda_backend | CUDA context |
| 21.2 | Done | phase_21_cuda_backend | CUDA GEMM |
| 21.3 | Done | phase_21_cuda_backend | CUDA attention |
| 21.4 | Done | phase_21_cuda_backend | CUDA pipeline |
| 22.1 | Done | phase_22_metal_backend | Metal device |
| 22.2 | Done | phase_22_metal_backend | Metal shaders |
| 22.3 | Done | phase_22_metal_backend | Metal pipeline |
| 23.1 | Done | phase_23_flashmla_moe | FlashMLA |
| 23.2 | Done | phase_23_flashmla_moe | MoE fused ops |
| 23.3 | Done | phase_23_flashmla_moe | Smart Expert Reduction |
| 24.1 | Done | phase_24_chat_templates_server | Jinja templates |
| 24.2 | Done | phase_24_chat_templates_server | HTTP server |
| 24.3 | Done | phase_24_chat_templates_server | OpenAI API |
| 25.1 | Done | phase_25_testing_infra | Unit tests |
| 25.2 | Done | phase_25_testing_infra | Integration tests |
| 25.3 | Done | phase_25_testing_infra | Benchmarks |
| 26.1 | Done | phase_26_multimodal_foundation | Vision encoder |
| 26.2 | Done | phase_26_multimodal_foundation | VL models |
| 27.1 | Pending | - | API docs |
| 27.2 | Pending | - | User docs |
| 27.3 | Pending | - | Dev docs |

---

*Created: March 17, 2026*
