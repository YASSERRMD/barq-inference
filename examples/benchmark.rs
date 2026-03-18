//! Performance benchmarking example
//!
//! Demonstrates how to:
//! - Benchmark token generation speed
//! - Compare different quantization levels
//! - Profile memory usage
//! - Test different batch sizes

use barq_inference::client::Client;
use barq_inference::models::ModelLoader;
use barq_core::testing::{BenchmarkTimer, TestStats};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Barq Inference - Performance Benchmarks ===\n");

    let loader = ModelLoader::new();
    let model_path = "path/to/model.gguf";
    let model = loader.load_from_file(model_path).await?;
    let client = Client::new(model)?;

    // ========================================================================
    // Benchmark 1: Tokens per Second
    // ========================================================================

    println!("1. Tokens per Second Benchmark\n");

    let prompt = "Write a detailed explanation of how neural networks work.";
    let num_runs = 5;
    let mut tps_stats = TestStats::new();

    for run in 0..num_runs {
        let (response, time) = BenchmarkTimer::measure(|| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    client.generate(
                        prompt,
                        &Default::default()
                    ).await
                })
            })
        });

        let tps = response.text.len() as f64 / time;
        tps_stats.add(tps);

        println!("  Run {}: {:.2} tokens/sec", run + 1, tps);
    }

    println!("\nTokens/sec Statistics:");
    println!("  Mean: {:.2}", tps_stats.mean());
    println!("  Median: {:.2}", tps_stats.median());
    println!("  Std Dev: {:.2}", tps_stats.std_dev());
    println!("  Min: {:.2}", tps_stats.min());
    println!("  Max: {:.2}", tps_stats.max());

    // ========================================================================
    // Benchmark 2: Memory Usage
    // ========================================================================

    println!("\n2. Memory Usage Benchmark\n");

    let memory_before = get_memory_usage();

    let long_prompt = "Continue this story: " & "Once upon a time, ".repeat(100);
    let config = barq_inference::client::GenerationConfig {
        max_tokens: 500,
        ..Default::default()
    };

    client.generate(&long_prompt, &config).await?;

    let memory_after = get_memory_usage();
    let memory_used = memory_after - memory_before;

    println!("  Memory before: {:.2} MB", memory_before);
    println!("  Memory after: {:.2} MB", memory_after);
    println!("  Memory used: {:.2} MB", memory_used);

    // ========================================================================
    // Benchmark 3: Batch Processing Speedup
    // ========================================================================

    println!("\n3. Batch Processing Speedup\n");

    let batch_sizes = vec![1, 2, 4, 8];
    let prompts_per_batch = 10;

    for batch_size in batch_sizes {
        let prompts: Vec<String> = (0..prompts_per_batch)
            .map(|i| format!("Tell me a joke number {}.", i))
            .collect();

        let (_, total_time) = BenchmarkTimer::measure(|| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    client.generate_batch(prompts.clone(), &Default::default()).await
                })
            })
        });

        let time_per_prompt = total_time / prompts_per_batch as f64;
        let speedup = 1.0 / time_per_prompt; // Normalized to batch_size=1

        println!("  Batch size {}: {:.3}ms/prompt (speedup: {:.2}x)",
            batch_size, time_per_prompt * 1000.0, speedup);
    }

    // ========================================================================
    // Benchmark 4: Different Quantization Levels
    // ========================================================================

    println!("\n4. Quantization Comparison\n");

    // This would require loading multiple models with different quantization
    // For demonstration, we'll show the structure

    let quantization_levels = vec![
        ("Q4_K_M", "4-bit K-quants medium"),
        ("Q5_K_M", "5-bit K-quants medium"),
        ("Q8_0", "8-bit quantization"),
        ("F16", "16-bit float"),
    ];

    println!("  Quantization levels and their expected characteristics:");
    for (name, description) in quantization_levels {
        println!("    - {}: {}", name, description);
    }

    // ========================================================================
    // Benchmark 5: Time to First Token (TTFT)
    // ========================================================================

    println!("\n5. Time to First Token\n");

    let ttft_runs = 10;
    let mut ttft_stats = TestStats::new();

    for run in 0..ttft_runs {
        let (result, time) = BenchmarkTimer::measure(|| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    client.generate_stream("Quick response:", &Default::default()).await
                })
            })
        });

        // Get first token
        let first_token_time = match result {
            Ok(mut stream) => {
                match stream.next().await {
                    Ok(Some(token)) => time,
                    _ => continue,
                }
            }
            _ => continue,
        };

        ttft_stats.add(first_token_time);
    }

    println!("\nTTFT Statistics ({} runs):", ttft_runs);
    println!("  Mean: {:.3}ms", ttft_stats.mean() * 1000.0);
    println!("  Median: {:.3}ms", ttft_stats.median() * 1000.0);
    println!("  P95: {:.3}ms", ttft_stats.percentile(95.0) * 1000.0);
    println!("  P99: {:.3}ms", ttft_stats.percentile(99.0) * 1000.0);

    Ok(())
}

/// Get current memory usage in MB
fn get_memory_usage() -> f64 {
    // This is a placeholder - actual implementation would use
    // platform-specific memory APIs
    #[cfg(unix)]
    {
        use std::process::Command;
        let output = Command::new("ps")
            .arg("-o")
            .arg("rss=")
            .arg(std::process::id().to_string())
            .output()
            .ok();

        if let Ok(output) {
            let rss_kb = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<f64>()
                .unwrap_or(0.0);
            return rss_kb / 1024.0; // Convert to MB
        }
    }

    0.0 // Fallback
}
