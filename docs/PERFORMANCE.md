# Performance Optimization Guide

## Table of Contents

1. [Tuning Strategy](#tuning-strategy)
2. [Recommended Profiles](#recommended-profiles)
3. [Quantization Choices](#quantization-choices)
4. [Benchmarking](#benchmarking)
5. [Practical Tips](#practical-tips)

## Tuning Strategy

Performance depends on four variables:

1. Model size and quantization
2. Context length
3. Backend selection
4. Batch size and sampling settings

The fastest way to improve throughput is usually:

1. Pick a smaller quantization
2. Use GPU or Metal offload when available
3. Keep the prompt window smaller than the model maximum
4. Batch prompts when possible

## Recommended Profiles

### GPU / Metal Throughput

```rust
use std::sync::Arc;
use models::context::{ContextParams, ModelContext};
use models::loader::Model;
use models::transformer::LlamaTransformer;

# async fn build(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
let model = Arc::new(Model::load(model_path).await?);
let transformer = Arc::new(LlamaTransformer::new(model.clone())?);
let params = ContextParams::gpu_optimized();
let _ctx = ModelContext::new(model, params, transformer)?;
# Ok(())
# }
```

Use this profile when you want higher throughput and the model fits on GPU memory.

### CPU Efficiency

```rust
let params = models::context::ContextParams::cpu_optimized();
```

Use this profile when you want predictable CPU-only inference or when GPU memory is limited.

### Quality First

```rust
let params = models::context::ContextParams::quality();
```

Choose a higher-quality quantization such as Q8_0 when memory is not the limiting factor.

### Speed First

```rust
let params = models::context::ContextParams::speed();
```

This profile lowers the context size and favors latency.

## Quantization Choices

Use the smallest quantization that still preserves acceptable output quality:

- `Q8_0` for quality-sensitive workloads
- `Q5_K` for balanced quality and throughput
- `Q4_K` for fast local inference
- `Q2_K` when memory pressure is the primary constraint

As a rule of thumb:

- Smaller quantization reduces memory use and usually increases throughput
- Larger quantization improves quality but costs more RAM and bandwidth
- Quantization choice often matters more than small parameter tweaks

## Benchmarking

### CLI Benchmarks

```bash
barq-inference benchmark -m model.gguf --iterations 10 --prompt-length 512 --gen-length 128
```

### Criterion Benchmarks

```bash
cargo bench -p barq-inference --benches --no-run
```

### Integration Checks

```bash
cargo test -p barq-inference --test integration_tests --quiet
```

Use the benchmark suite to compare:

- Different quantizations
- Different prompt lengths
- Different context sizes
- CPU versus GPU versus Metal

## Practical Tips

- Keep `--context-size` as small as your task allows.
- Prefer batch processing for repeated prompts.
- Use speculative decoding for low-latency generation when a draft model is available.
- Use JSON mode only when structured output is required.
- On Apple Silicon, check `barq-inference metal-info` before tuning GPU offload.
- On systems with limited VRAM, lower the number of GPU layers before reducing model quality.

## Related Docs

- [User Guide](./USER_GUIDE.md)
- [Migration Guide](./MIGRATION.md)
- [Contributing](../CONTRIBUTING.md)
- [README](../README.md)
