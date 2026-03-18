# Barq Inference - User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- Rust 1.70 or later
- Cargo
- For CUDA support: NVIDIA GPU with CUDA 11.0+
- For Metal support: macOS 12.0+ with Apple Silicon

### From Source

```bash
# Clone the repository
git clone https://github.com/YASSERRMD/barq-inference.git
cd barq-inference

# Build with default features (CPU only)
cargo build --release

# Build with CUDA support
cargo build --release --features cuda

# Build with Metal support (macOS)
cargo build --release --features metal

# Install binary
cargo install --path .
```

### Cargo.toml

Add to your `Cargo.toml`:

```toml
[dependencies]
barq-inference = { version = "0.1", features = ["cuda"] } # or ["metal"] for Apple Silicon
```

## Quick Start

### Basic Text Generation

```rust
use barq_inference::client::Client;
use barq_inference::models::ModelLoader;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a model
    let loader = ModelLoader::new();
    let model = loader.load_from_file("model.gguf").await?;

    // Create client
    let client = Client::new(model)?;

    // Generate text
    let response = client.generate(
        "Once upon a time",
        &Default::default()
    ).await?;

    println!("{}", response.text);
    Ok(())
}
```

### Streaming Generation

```rust
use barq_inference::client::Client;
use barq_inference::models::ModelLoader;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let loader = ModelLoader::new();
    let model = loader.load_from_file("model.gguf").await?;
    let client = Client::new(model)?;

    let mut stream = client.generate_stream(
        "Continue the story:",
        &Default::default()
    ).await?;

    while let Some(token) = stream.next().await? {
        print!("{}", token.text);
    }

    Ok(())
}
```

## Configuration

### Generation Parameters

```rust
use barq_inference::client::GenerationConfig;

let config = GenerationConfig {
    max_tokens: 100,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repetition_penalty: 1.1,
    frequency_penalty: 0.5,
    presence_penalty: 0.5,
    seed: Some(42),
    ..Default::default()
};

let response = client.generate(prompt, &config).await?;
```

### Parameter Descriptions

- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Sampling temperature (0.0 - 2.0, lower = more deterministic)
- `top_p`: Nucleus sampling threshold (0.0 - 1.0)
- `top_k`: Top-k sampling (keep k most likely tokens)
- `repetition_penalty`: Penalty for repeating tokens (1.0 = no penalty)
- `frequency_penalty`: Penalty for token frequency
- `presence_penalty`: Penalty for token presence
- `seed`: Random seed for reproducible generation

## Usage Examples

### JSON Mode (Structured Output)

```rust
use barq_inference::grammar::GrammarCompiler;

let json_schema = r#"
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  },
  "required": ["name", "age"]
}
"#;

let compiler = GrammarCompiler::new();
let grammar = compiler.compile_json(json_schema)?;

let config = GenerationConfig {
    grammar: Some(grammar),
    ..Default::default()
};

let response = client.generate(
    "Generate a person profile:",
    &config
).await?;
```

### Custom Sampling Pipeline

```rust
use barq_inference::sampling::{SamplerChain, Temperature, TopK, TopP};

let mut sampler = SamplerChain::new();
sampler.add(Box::new(Temperature::new(0.8)));
sampler.add(Box::new(TopK::new(40)));
sampler.add(Box::new(TopP::new(0.95)));

let config = GenerationConfig {
    sampler: Some(sampler),
    ..Default::default()
};
```

### Batch Processing

```rust
let prompts = vec![
    "What is AI?",
    "Explain machine learning.",
    "What is deep learning?",
];

let responses = client.generate_batch(
    prompts,
    &Default::default()
).await?;

for response in responses {
    println!("{}", response.text);
}
```

## API Reference

### Client

Main interface for model inference.

#### Methods

- `new(model: Model) -> Result<Client, Error>`
- `generate(&self, prompt: &str, config: &GenerationConfig) -> Result<Response, Error>`
- `generate_stream(&self, prompt: &str, config: &GenerationConfig) -> Result<TokenStream, Error>`
- `generate_batch(&self, prompts: Vec<String>, config: &GenerationConfig) -> Result<Vec<Response>, Error>`

### GenerationConfig

Configuration for text generation.

#### Fields

- `max_tokens: usize` - Default: 512
- `temperature: f32` - Default: 0.7
- `top_p: f32` - Default: 0.9
- `top_k: usize` - Default: 40
- `repetition_penalty: f32` - Default: 1.0
- `frequency_penalty: f32` - Default: 0.0
- `presence_penalty: f32` - Default: 0.0
- `grammar: Option<Grammar>` - Default: None
- `sampler: Option<Box<dyn Sampler>>` - Default: None
- `callback: Option<TokenCallback>` - Default: None
- `seed: Option<u64>` - Default: None

### Response

Generated text response.

#### Fields

- `text: String` - Generated text
- `prompt: String` - Original prompt
- `tokens_generated: usize` - Number of tokens generated
- `finish_reason: FinishReason` - Why generation stopped
- `timing: TimingInfo` - Performance information

## Performance Tuning

### Model Selection

| Quantization | Memory Usage | Speed | Quality |
|--------------|--------------|-------|--------|
| Q4_K_M | ~4GB | Fastest | Good |
| Q5_K_M | ~5GB | Fast | Better |
| Q8_0 | ~8GB | Medium | Excellent |
| F16 | ~16GB | Slow | Perfect |

### Batch Size Optimization

```rust
// Experiment with batch sizes for better GPU utilization
let batch_sizes = vec![1, 2, 4, 8, 16];

for batch_size in batch_sizes {
    let start = std::time::Instant::now();
    let responses = client.generate_batch(prompts.clone(), &config).await?;
    let elapsed = start.elapsed();

    let tps = responses.iter()
        .map(|r| r.tokens_generated)
        .sum::<usize>() as f64 / elapsed.as_secs_f64();

    println!("Batch size {}: {:.2} tokens/sec", batch_size, tps);
}
```

### Memory Optimization

```rust
// Enable CPU offloading for large models
let config = ModelConfig {
    n_gpu_layers: 20, // Number of layers to keep on GPU
    ..Default::default()
};

let model = loader.load_with_config(model_path, config).await?;
```

## Troubleshooting

### Out of Memory

```rust
// Reduce number of GPU layers
let config = ModelConfig {
    n_gpu_layers: 10, // Reduce from default
    ..Default::default()
};

// Or use CPU-only mode
let model = loader.load_from_file_cpu(model_path).await?;
```

### Slow Generation

```rust
// 1. Check if GPU is being used
println!("Device: {:?}", client.device_info());

// 2. Increase batch size for better throughput
let config = GenerationConfig {
    ..Default::default()
};
let responses = client.generate_batch(prompts, &config).await?;

// 3. Use lower precision quantization
let model = loader.load_quantized(model_path, QuantizationType::Q4_K).await?;
```

### Poor Quality Output

```rust
// 1. Adjust sampling parameters
let config = GenerationConfig {
    temperature: 0.7,      // Lower for more deterministic
    top_p: 0.9,             // Nucleus sampling
    top_k: 40,              // Top-k sampling
    repetition_penalty: 1.1, // Reduce repetition
    ..Default::default()
};

// 2. Use better prompt engineering
let prompt = "Question: What is the capital of France?\nAnswer:";

// 3. Use JSON mode for structured output
let grammar = compiler.compile_json(json_schema)?;
let config = GenerationConfig {
    grammar: Some(grammar),
    ..Default::default()
};
```

### CUDA Errors

```rust
// Ensure CUDA is available
if let Ok(count) = barq_inference::backend::CudaBackend::device_count() {
    println!("Found {} CUDA devices", count);
}

// Check device capabilities
let backend = barq_inference::backend::CudaBackend::new(0)?;
println!("Device: {}", backend.props().name);
println!("Compute Capability: {:?}",
    backend.props().compute_capability);
println!("Total Memory: {} MB",
    backend.props().total_memory / (1024 * 1024));
```

## Additional Resources

- [API Documentation](https://docs.rs/barq-inference)
- [Examples](https://github.com/YASSERRMD/barq-inference/tree/main/examples)
- [Performance Guide](./PERFORMANCE.md)
- [Contributing](./CONTRIBUTING.md)
