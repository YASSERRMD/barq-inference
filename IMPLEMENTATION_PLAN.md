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
- [ ] Create `models/src/arch/qwen.rs` for Qwen base
- [ ] Create `models/src/arch/qwen2.rs` for Qwen2
- [ ] Create `models/src/arch/qwen3.rs` for Qwen3
- [ ] Implement RoPE with NTK-aware scaling
- [ ] Add GQA support for Qwen models
- [ ] Add test GGUF files

### Phase 18.2: DeepSeek Architecture
- [ ] Create `models/src/arch/deepseek.rs`
- [ ] Implement Multi-head Latent Attention (MLA)
- [ ] Add MoE routing for DeepSeek-MoE
- [ ] Implement FFN with SwiGLU
- [ ] Add test cases

### Phase 18.3: Mistral/Mixtral Improvements
- [ ] Complete `models/src/mistral.rs` implementation
- [ ] Complete `models/src/mixtral.rs` with full MoE routing
- [ ] Add sliding window attention
- [ ] Implement expert load balancing
- [ ] Add benchmarking

### Phase 18.4: Architecture Registry
- [ ] Create `models/src/arch_registry.rs`
- [ ] Map GGUF architecture names to implementations
- [ ] Add architecture detection from metadata
- [ ] Create unified `LlmArch` trait
- [ ] Document all supported architectures

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
- [ ] Add AVX-512 detection in `barq_core/src/platform.rs`
- [ ] Implement AVX-512 matrix multiplication
- [ ] Implement AVX-512 quantization kernels
- [ ] Add runtime dispatch based on CPU features
- [ ] Benchmark vs AVX2 baseline

### Phase 20.2: ARM NEON Improvements
- [ ] Audit existing NEON code
- [ ] Add missing NEON intrinsics
- [ ] Implement NEON GEMM kernels
- [ ] Test on Apple Silicon
- [ ] Document performance

### Phase 20.3: GEMM Optimizations
- [ ] Implement blocked GEMM in `barq_core/src/gemm.rs`
- [ ] Add cache-aware tiling
- [ ] Implement packing for cache efficiency
- [ ] Add threading with rayon
- [ ] Benchmark different sizes

### Phase 20.4: Prompt Processing Optimization
- [ ] Optimize prompt encoding path
- [ ] Implement parallel prompt processing
- [ ] Add batched prompt caching
- [ ] Measure TTFT improvements
- [ ] Document results

---

## Phase 21: CUDA Backend (P0)

**Goal:** Enable NVIDIA GPU inference.

### Phase 21.1: CUDA Context and Memory
- [ ] Create `backend/src/cuda/context.rs`
- [ ] Implement CUDA device detection
- [ ] Implement GPU memory allocation
- [ ] Add CUDA error handling
- [ ] Test device initialization

### Phase 21.2: CUDA GEMM Kernels
- [ ] Create `backend/src/cuda/kernels.rs`
- [ ] Implement cuBLAS integration
- [ ] Add quantization CUDA kernels
- [ ] Implement matrix operations
- [ ] Benchmark vs CPU

### Phase 21.3: CUDA Attention
- [ ] Implement Flash Attention CUDA
- [ ] Add KV cache GPU management
- [ ] Implement RoPE on GPU
- [ ] Add softmax kernel
- [ ] Test with model

### Phase 21.4: CUDA Pipeline Integration
- [ ] Create `backend/src/cuda/pipeline.rs`
- [ ] Implement layer-by-layer GPU execution
- [ ] Add CPU fallback path
- [ ] Implement tensor transfers
- [ ] End-to-end testing

---

## Phase 22: Metal Backend (P1)

**Goal:** Enable Apple Silicon GPU inference.

### Phase 22.1: Metal Device Setup
- [ ] Create `backend/src/metal/device.rs`
- [ ] Implement Metal device detection
- [ ] Create Metal command queue
- [ ] Add buffer management
- [ ] Test on macOS

### Phase 22.2: Metal Shaders
- [ ] Create Metal shader files (`.metal`)
- [ ] Implement GEMM shader
- [ ] Implement attention shader
- [ ] Implement quantization shaders
- [ ] Compile shaders at build time

### Phase 22.3: Metal Integration
- [ ] Create `backend/src/metal/pipeline.rs`
- [ ] Implement tensor operations
- [ ] Add KV cache Metal buffer
- [ ] Integrate with model inference
- [ ] Benchmark on Apple Silicon

---

## Phase 23: FlashMLA and MoE Optimizations (P2)

**Goal:** Optimize DeepSeek and MoE model inference.

### Phase 23.1: FlashMLA Implementation
- [ ] Create `advanced/src/flash_mla.rs`
- [ ] Implement MLA-2 (K-cache only)
- [ ] Implement MLA-3 (optimized)
- [ ] Add `-mla` CLI flag
- [ ] Document usage

### Phase 23.2: MoE Fused Operations
- [ ] Create `models/src/moe_fused.rs`
- [ ] Implement fused FFN up/gate
- [ ] Implement expert batch processing
- [ ] Add `-fmoe` CLI flag
- [ ] Benchmark MoE speedup

### Phase 23.3: Smart Expert Reduction
- [ ] Implement SER for DeepSeek
- [ ] Add `-ser` CLI flag
- [ ] Add expert routing optimization
- [ ] Test accuracy vs speed
- [ ] Document trade-offs

---

## Phase 24: Chat Templates and Server (P2)

**Goal:** Production-ready serving capabilities.

### Phase 24.1: Jinja Chat Templates
- [ ] Create `vocab/src/chat_template.rs`
- [ ] Implement Jinja2 template parser
- [ ] Add common templates (LLaMA, Mistral, Qwen, etc.)
- [ ] Implement template application
- [ ] Test with various models

### Phase 24.2: HTTP Server
- [ ] Create `cli/src/server.rs`
- [ ] Implement `/v1/chat/completions` endpoint
- [ ] Implement `/v1/completions` endpoint
- [ ] Add streaming support (SSE)
- [ ] Add CORS support

### Phase 24.3: OpenAI API Compatibility
- [ ] Implement `/v1/models` endpoint
- [ ] Implement `/v1/responses` (ik_llama.cpp style)
- [ ] Add token counting
- [ ] Add rate limiting
- [ ] Document API compatibility

---

## Phase 25: Testing Infrastructure (P1)

**Goal:** Ensure correctness and prevent regressions.

### Phase 25.1: Unit Test Expansion
- [ ] Add tests for each quantization type
- [ ] Add tests for each architecture
- [ ] Add tests for sampling algorithms
- [ ] Add tests for attention mechanisms
- [ ] Achieve >80% coverage

### Phase 25.2: Integration Tests
- [ ] Create `tests/integration/`
- [ ] Add model loading tests
- [ ] Add end-to-end inference tests
- [ ] Add memory leak tests
- [ ] Add CI workflow

### Phase 25.3: Benchmark Suite
- [ ] Create `benches/` with criterion benchmarks
- [ ] Add token generation benchmarks
- [ ] Add prompt processing benchmarks
- [ ] Add memory benchmarks
- [ ] Track performance over time

---

## Phase 26: Multimodal Foundation (P3)

**Goal:** Enable vision-language model support.

### Phase 26.1: Vision Encoder Interface
- [ ] Create `models/src/vision/mod.rs`
- [ ] Define vision encoder trait
- [ ] Add CLIP-like encoder support
- [ ] Add image preprocessing
- [ ] Test embedding extraction

### Phase 26.2: Vision-Language Models
- [ ] Create `models/src/arch/qwen2vl.rs`
- [ ] Create `models/src/arch/llava.rs`
- [ ] Implement cross-attention
- [ ] Add image token handling
- [ ] Test with multimodal GGUF

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
| 18.2 | Pending | - | DeepSeek architecture |
| 18.3 | Pending | - | Mistral improvements |
| 18.4 | Done | main | Architecture registry |
| 19.1 | Done | phase_19_grammar_json | Grammar parser |
| 19.2 | Done | phase_19_grammar_json | Grammar sampling |
| 19.3 | Done | phase_19_grammar_json | JSON mode |
| 20.1 | Pending | - | AVX-512 kernels |
| 20.2 | Pending | - | NEON improvements |
| 20.3 | Pending | - | GEMM optimizations |
| 20.4 | Pending | - | Prompt optimization |
| 21.1 | Pending | - | CUDA context |
| 21.2 | Pending | - | CUDA GEMM |
| 21.3 | Pending | - | CUDA attention |
| 21.4 | Pending | - | CUDA pipeline |
| 22.1 | Pending | - | Metal device |
| 22.2 | Pending | - | Metal shaders |
| 22.3 | Pending | - | Metal pipeline |
| 23.1 | Pending | - | FlashMLA |
| 23.2 | Pending | - | MoE fused ops |
| 23.3 | Pending | - | Smart Expert Reduction |
| 24.1 | Pending | - | Jinja templates |
| 24.2 | Pending | - | HTTP server |
| 24.3 | Pending | - | OpenAI API |
| 25.1 | Pending | - | Unit tests |
| 25.2 | Pending | - | Integration tests |
| 25.3 | Pending | - | Benchmarks |
| 26.1 | Pending | - | Vision encoder |
| 26.2 | Pending | - | VL models |
| 27.1 | Pending | - | API docs |
| 27.2 | Pending | - | User docs |
| 27.3 | Pending | - | Dev docs |

---

*Created: March 17, 2026*
