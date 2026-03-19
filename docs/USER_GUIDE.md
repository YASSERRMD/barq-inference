# Barq Inference - User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Flags](#cli-flags)
4. [Rust API](#rust-api)
5. [Multimodal Support](#multimodal-support)
6. [Troubleshooting](#troubleshooting)
7. [Additional Resources](#additional-resources)

## Installation

### From Source

```bash
git clone https://github.com/YASSERRMD/barq-inference.git
cd barq-inference
cargo build --release
cargo install --path cli
```

### GPU Builds

```bash
# NVIDIA CUDA
cargo build --release --features cuda

# Apple Silicon Metal
cargo build --release --features metal
```

## Quick Start

### Run a Model

```bash
barq-inference run -m model.gguf -p "Explain rotary position embeddings" --max-tokens 128
```

### Chat Mode

```bash
barq-inference chat -m model.gguf --system-prompt "Be concise."
```

### OpenAI-Compatible Server

```bash
barq-inference http-server -m model.gguf --host 0.0.0.0 --port 8000 --rate-limit-rpm 60
```

The server exposes:

- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`

Streaming responses use Server-Sent Events and include token usage metadata.

## CLI Flags

The most useful `run` flags are:

| Flag | Purpose |
|------|---------|
| `--temperature` | Controls randomness in sampling |
| `--top-k` | Keeps the `k` highest-probability tokens |
| `--top-p` | Applies nucleus sampling |
| `--context-size` | Overrides the prompt context window |
| `--json` | Forces JSON mode with grammar validation |
| `--speculative` | Enables speculative decoding |
| `--draft-max` | Controls draft-model speculation depth |
| `--mla` | Enables FlashMLA context expansion for DeepSeek-family models |
| `--fmoe` | Enables fused MoE dispatch helpers |
| `--ser` | Enables smart expert reduction |

Global flags include:

- `--verbose`
- `--threads`
- `--cuda-graphs`
- `--flash-attn`
- `--preset`

Platform diagnostics:

- `barq-inference metal-info`
- `barq-inference wasm-info`

## Rust API

```rust
use std::sync::Arc;

use models::context::{Batch, ContextParams, ModelContext};
use models::loader::Model;
use models::transformer::LlamaTransformer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("model.gguf").await?);
    let transformer = Arc::new(LlamaTransformer::new(model.clone())?);
    let ctx = ModelContext::with_model(model, transformer)?;

    let batch = Batch::from_tokens(&[1, 2, 3, 4]);
    let logits = ctx.encode(&batch).await?;
    println!("logits: {}", logits.len());

    let params = ContextParams::gpu_optimized();
    println!("flash attention: {}", params.flash_attn);

    Ok(())
}
```

### Token Sampling

```rust
use models::context::ModelContext;

fn choose_token(ctx: &ModelContext, logits: &[f32]) -> i32 {
    ctx.sample(logits, 0.8, 40, 0.95).unwrap()
}
```

## Multimodal Support

Barq now includes a vision foundation for future multimodal models:

- `models::ImageInput` for raw RGB images
- `models::ImagePreprocessor` for resize and normalization
- `models::ClipVisionEncoder` for deterministic patch embeddings
- `models::Qwen2VlModel` and `models::LlavaModel` for model wrappers

The current CLI does not yet accept image files directly, but the Rust API is ready for multimodal adapters.

## Troubleshooting

### Model Fails To Load

- Check that the GGUF metadata contains a supported architecture name.
- Confirm the model uses the expected quantization and tensor layout.
- Use `barq-inference info -m model.gguf` to inspect the file metadata.

### Out Of Memory

- Reduce `--context-size`.
- Use a smaller quantization such as Q4_K or Q5_K.
- Prefer `ContextParams::cpu_optimized()` when GPU memory is tight.

### Slow Generation

- Use `ContextParams::gpu_optimized()` on supported hardware.
- Try a smaller batch size or a more aggressive quantization.
- Benchmark with `barq-inference benchmark -m model.gguf`.

### JSON Output Is Rejected

- Make sure the prompt is compatible with JSON mode.
- Keep the schema simple when starting out.
- Inspect the generated text for trailing commentary that breaks validation.

## Additional Resources

- [Performance Guide](./PERFORMANCE.md)
- [Migration Guide](./MIGRATION.md)
- [Contributing](../CONTRIBUTING.md)
- [README](../README.md)
