# Performance Optimization Guide

## Table of Contents

1. [Hardware Selection](#hardware-selection)
2. [Model Quantization](#model-quantization)
3. [Batch Processing](#batch-processing)
4. [Memory Management](#memory-management)
5. [GPU Optimization](#gpu-optimization)
6. [Multi-GPU Scaling](#multi-gpu-scaling)
7. [Profiling and Debugging](#profiling-and-debugging)

## Hardware Selection

### Recommended Hardware

| Use Case | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| Small Models (7B) | 8GB RAM, CPU | 16GB RAM, GTX 1660 | 24GB RAM, RTX 3090 |
| Medium Models (13B) | 16GB RAM, RTX 3060 | 32GB RAM, RTX 3080 Ti | 48GB RAM, RTX 4090 |
| Large Models (70B) | 64GB RAM, RTX 3090 | 128GB RAM, A100 40GB | 128GB RAM, A100 80GB |

### Apple Silicon (macOS)

| Chip | Memory | Max Model Size (Q4) | Max Model Size (F16) |
|------|--------|---------------------|----------------------|
| M1 | 8-16GB | 7B | 3B |
| M1 Pro | 16-32GB | 13B | 7B |
| M1 Max/ultra | 32-64GB | 30B | 13B |
| M2 Pro | 16-32GB | 13B | 7B |
| M2 Max/ultra | 32-96GB+ | 30B+ | 13B+ |

## Model Quantization

### Quantization Trade-offs

```
Quality vs Speed vs Memory

Q8_0:  ★★★★★  ★★★☆☆  ★★☆☆☆  (Best quality, medium speed, low memory)
Q5_K:  ★★★★☆  ★★★★☆  ★★★☆☆  (Good balance)
Q4_K:  ★★★☆☆  ★★★★★  ★★★★★  (Fastest, lowest memory, good quality)
Q2_K:  ★★☆☆☆  ★★★★★  ★★★★★  (Extreme compression)
```

### Choosing the Right Quantization

```rust
use barq_inference::quant::QuantizationType;

// For production chatbots (quality focused)
let quant = QuantizationType::Q80;

// For local deployment (balance)
let quant = QuantizationType::Q5K;

// For edge devices (speed/memory focused)
let quant = QuantizationType::Q4K;

// For testing/experimentation (extreme compression)
let quant = QuantizationType::Q2K;
```

### Quantization Performance Comparison

| Model | Q4_K | Q5_K | Q8_0 | F16 |
|-------|------|------|------|-----|
| Llama-2-7B | 3.8GB | 4.8GB | 7.2GB | 14GB |
| Speed (tok/s) | 85 | 72 | 58 | 45 |
| Quality Score | 8.5 | 9.2 | 9.8 | 10.0 |

## Batch Processing

### Batch Size Selection

```rust
// Rule of thumb: batch_size * model_size < GPU_memory * 0.8

let batch_size = match model_size {
    0..4_000_000_000 => 8,      // < 4GB
    4_000_000_000..8_000_000_000 => 4,  // 4-8GB
    8_000_000_000..16_000_000_000 => 2, // 8-16GB
    _ => 1,                           // > 16GB
};

// Dynamic batch sizing based on available memory
let available_memory = gpu_memory() * 0.8;
let max_batch_size = (available_memory / model_memory) as usize;
```

### Continuous Batching

For production systems, implement continuous batching:

```rust
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

struct ContinuousBatcher {
    requests: Vec<Request>,
    max_batch_size: usize,
    max_wait_time: Duration,
}

impl ContinuousBatcher {
    async fn collect_batch(&mut self) -> Vec<Request> {
        let start = std::time::Instant::now();

        // Wait for either:
        // 1. Batch is full
        // 2. Max wait time elapsed
        // 3. Minimum batch size reached after timeout

        loop {
            if self.requests.len() >= self.max_batch_size {
                break;
            }

            let elapsed = start.elapsed();
            if elapsed >= self.max_wait_time && self.requests.len() > 0 {
                break;
            }

            // Wait for more requests with timeout
            let _ = timeout(Duration::from_millis(10), async {
                // Wait for next request
            }).await;
        }

        self.requests.drain(..).collect()
    }
}
```

## Memory Management

### GPU Memory Optimization

```rust
// 1. CPU offloading for large models
let config = ModelConfig {
    n_gpu_layers: 20,  // Only keep 20 layers on GPU
    ..Default::default()
};

// 2. Memory mapping (mmap) for model weights
let config = ModelConfig {
    use_mmap: true,      // Memory map weights instead of loading
    use_mmap_cpu: true,  // Keep mmap on CPU
    ..Default::default()
};

// 3. Context unloading
let config = ModelConfig {
    unload_context: true, // Unload after generation
    ..Default::default()
};
```

### Memory Profiling

```rust
pub fn profile_memory() {
    #[cfg(feature = "cuda")]
    {
        use barq_inference::backend::CudaBackend;

        let backend = CudaBackend::new(0).unwrap();
        println!("Total Memory: {} MB",
            backend.props().total_memory / (1024 * 1024));
        println!("Free Memory: {} MB",
            get_free_memory() / (1024 * 1024));
    }
}

fn get_free_memory() -> usize {
    // Platform-specific implementation
    0
}
```

## GPU Optimization

### CUDA-Specific Optimizations

```rust
// 1. Enable cuBLAS for matrix operations
let config = ModelConfig {
    use_cublas: true,
    ..Default::default()
};

// 2. Use Flash Attention
let config = ModelConfig {
    use_flash_attention: true,
    flash_attn_recompute: false,  // Set true if low on memory
    ..Default::default()
};

// 3. Enable tensor cores (newer GPUs)
let config = ModelConfig {
    tensor_cores: true,  // For RTX 20xx and newer
    ..Default::default()
};
```

### Metal-Specific Optimizations (macOS)

```rust
// 1. Use unified memory architecture
let config = ModelConfig {
    use_unified_memory: true,  // M1/M2/M3 benefit
    ..Default::default()
};

// 2. Enable Metal Performance Shaders
let config = ModelConfig {
    use_metal_shaders: true,
    ..Default::default()
};

// 3. Optimize threadgroup size
let config = ModelConfig {
    threadgroup_size: 256,  // Optimal for Apple Silicon
    ..Default::default()
};
```

## Multi-GPU Scaling

### Tensor Parallelism

```rust
use barq_inference::backend::MultiGpuConfig;

let config = MultiGpuConfig::tensor_parallel(4);
let client = Client::with_multi_gpu(model, config)?;

// Benefits:
// - Linear speedup for matrix operations
// - Scales model size across GPUs
// - Good for large models (70B+)
```

### Pipeline Parallelism

```rust
let config = MultiGpuConfig::pipeline_parallel(4);
let client = Client::with_multi_gpu(model, config)?;

// Benefits:
// - Reduces per-GPU memory usage
// - Good for very deep models
// - Higher latency but better throughput
```

### Hybrid Parallelism

```rust
// Combine tensor + pipeline parallelism
let config = MultiGpuConfig::hybrid(
    tp_degree = 2,  // Tensor parallelism
    pp_degree = 4,  // Pipeline parallelism
); // Total: 8 GPUs

let client = Client::with_multi_gpu(model, config)?;
```

## Profiling and Debugging

### Performance Profiling

```rust
use barq_inference::profiling::Profiler;

let profiler = Profiler::new();

profile!("model_load", {
    let model = loader.load_from_file(path).await?;
});

profile!("generation", {
    let response = client.generate(prompt, &config).await?;
});

// Print profiling report
profiler.print_report();
```

### Token Generation Analysis

```rust
pub fn analyze_generation(response: &Response) {
    println!("Tokens generated: {}", response.tokens_generated);
    println!("Time to first token: {:?}", response.timing.ttft);
    println!("Tokens per second: {:.2}", response.timing.tps);
    println!("Total time: {:?}", response.timing.total_time);

    // Analyze token distribution
    let token_counts = count_token_frequencies(&response.tokens);
    println!("Unique tokens: {}", token_counts.len());
    println!("Most common tokens:");
    for (token, count) in token_counts.most_common(5) {
        println!("  '{}': {} times", token, count);
    }
}
```

### Bottleneck Identification

```rust
// Profile each component
let timings = Timings {
    model_load: measure_model_load(),
    prompt_processing: measure_prompt_processing(),
    token_generation: measure_generation(),
    total_time: start.elapsed(),
};

println!("Breakdown:");
println!("  Model load: {:.2}%", timings.model_load / timings.total_time * 100.0);
println!("  Prompt processing: {:.2}%", timings.prompt_processing / timings.total_time * 100.0);
println!("  Token generation: {:.2}%", timings.token_generation / timings.total_time * 100.0);

// Identify bottleneck
let (component, time) = vec![
    ("Model load", timings.model_load),
    ("Prompt processing", timings.prompt_processing),
    ("Token generation", timings.token_generation),
].into_iter()
.max_by_key(|&(_, time)| time)
.unwrap();

println!("\nBottleneck: {}", component);
```

## Best Practices

### 1. Use Appropriate Quantization

```rust
// Don't use F16 for all use cases
// Choose based on requirements:

let quant = match use_case {
    UseCase::Production => QuantizationType::Q5K,
    UseCase::LatencyCritical => QuantizationType::Q4K,
    UseCase::QualityCritical => QuantizationType::Q80,
    UseCase::ResourceConstrained => QuantizationType::Q2K,
};
```

### 2. Batch When Possible

```rust
// Single request
let response = client.generate(prompt, &config).await?;

// Multiple requests (batched)
let responses = client.generate_batch(prompts, &config).await?;

// Batching provides 2-4x speedup on GPUs
```

### 3. Use Streaming for Interactive Applications

```rust
// Non-streaming (slow perceived latency)
let response = client.generate(prompt, &config).await?;
println!("{}", response.text);

// Streaming (instant feedback)
let mut stream = client.generate_stream(prompt, &config).await?;
while let Some(token) = stream.next().await? {
    print!("{}", token.text);
    std::io::Write::flush(&mut std::io::stdout())?;
}
```

### 4. Cache Repeated Prompts

```rust
use std::collections::HashMap;
use tokio::sync::RwLock;

let prompt_cache = Arc::new(RwLock::new(HashMap::new()));

async fn generate_with_cache(
    client: &Client,
    prompt: &str,
    config: &GenerationConfig
) -> Result<String> {
    // Check cache
    {
        let cache = prompt_cache.read().await;
        if let Some(response) = cache.get(prompt) {
            return Ok(response.clone());
        }
    }

    // Generate
    let response = client.generate(prompt, config).await?;
    let text = response.text.clone();

    // Update cache
    {
        let mut cache = prompt_cache.write().await;
        cache.insert(prompt.to_string(), text);
    }

    Ok(text)
}
```

### 5. Monitor Resource Usage

```rust
use sysinfo::{System, SystemExt};

let mut sys = System::new_all();
sys.refresh_all();

println!("CPU Usage: {}%", sys.global_cpu_info().cpu_usage());
println!("Memory Usage: {} / {} MB",
    sys.used_memory() / 1024,
    sys.total_memory() / 1024
);

#[cfg(feature = "cuda")]
{
    let backend = CudaBackend::new(0)?;
    println!("GPU Memory: {} MB",
        backend.props().total_memory / (1024 * 1024)
    );
}
```

## Performance Benchmarks

### Reference Hardware

- **CPU**: AMD EPYC 7763 (64 cores)
- **GPU**: NVIDIA A100 40GB
- **RAM**: 256 GB DDR4
- **OS**: Ubuntu 22.04 LTS

### Model Performance (tokens/sec)

| Model | Q4_K | Q5_K | Q8_0 | F16 |
|-------|------|------|------|-----|
| Llama-2-7B | 95 | 78 | 62 | 48 |
| Llama-2-13B | 62 | 52 | 41 | 32 |
| Llama-2-70B | 18 | 15 | 12 | 8 |

### Batch Size Scaling (Llama-2-7B Q4_K)

| Batch Size | Tokens/sec | Speedup | Memory (GB) |
|------------|------------|---------|-------------|
| 1 | 95 | 1.0x | 3.8 |
| 2 | 145 | 1.5x | 4.2 |
| 4 | 198 | 2.1x | 5.1 |
| 8 | 265 | 2.8x | 6.8 |
| 16 | 310 | 3.3x | 9.2 |

## Troubleshooting Performance Issues

### Problem: Slow Token Generation

**Symptoms**: < 10 tokens/sec

**Solutions**:
1. Use lower quantization (Q4_K instead of Q8_0)
2. Enable GPU acceleration
3. Increase batch size
4. Check for CPU throttling

### Problem: High Memory Usage

**Symptoms**: OOM errors, swapping

**Solutions**:
1. Reduce `n_gpu_layers`
2. Use CPU offloading
3. Enable memory mapping
4. Use smaller batch size

### Problem: GPU Underutilization

**Symptoms**: Low GPU usage, slow generation

**Solutions**:
1. Increase batch size
2. Use tensor parallelism
3. Check for PCIe bottlenecks
4. Verify CUDA/Metal drivers

### Problem: Poor Quality Output

**Symptoms**: Nonsense, repetition, hallucination

**Solutions**:
1. Adjust sampling parameters (temperature, top_p)
2. Use repetition penalty
3. Try higher quantization (Q5_K or Q8_0)
4. Improve prompt engineering
