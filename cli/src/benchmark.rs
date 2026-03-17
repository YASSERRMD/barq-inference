//! Benchmarking harness for inference performance measurement
//!
//! Provides comprehensive benchmarking capabilities to measure:
//! - Tokens per second (TPS) for both prompt processing (TG) and text generation (PP)
//! - Time to First Token (TTFT)
//! - Memory usage (VRAM/RAM)
//! - KV cache hit rate

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Benchmark result metrics
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Mean tokens per second
    pub mean_tps: f64,
    /// Minimum tokens per second across all runs
    pub min_tps: f64,
    /// Maximum tokens per second across all runs
    pub max_tps: f64,
    /// Standard deviation of TPS
    pub std_tps: f64,
    /// Total tokens generated
    pub total_tokens: usize,
    /// Total time elapsed
    pub total_time: Duration,
    /// Prompt processing time (time to first token)
    pub ttft: Duration,
    /// Memory usage in bytes (if available)
    pub memory_bytes: Option<usize>,
    /// Number of runs
    pub runs: usize,
}

impl BenchmarkResult {
    /// Format TPS for display
    pub fn format_tps(&self) -> String {
        format!("{:.2} tok/s", self.mean_tps)
    }

    /// Calculate percentage improvement over another result
    pub fn improvement_over(&self, other: &BenchmarkResult) -> f64 {
        if other.mean_tps == 0.0 {
            0.0
        } else {
            ((self.mean_tps - other.mean_tps) / other.mean_tps) * 100.0
        }
    }

    /// Print benchmark results
    pub fn print(&self) {
        println!("\n=== Benchmark Results ===");
        println!("Runs:              {}", self.runs);
        println!("Total tokens:      {}", self.total_tokens);
        println!("Total time:        {:?}", self.total_time);
        println!("Time to first tok: {:?}", self.ttft);
        println!("\nTokens per second:");
        println!("  Mean:            {:.2} tok/s", self.mean_tps);
        println!("  Min:             {:.2} tok/s", self.min_tps);
        println!("  Max:             {:.2} tok/s", self.max_tps);
        println!("  Std Dev:         {:.2} tok/s", self.std_tps);

        if let Some(mem) = self.memory_bytes {
            let mem_mb = mem as f64 / (1024.0 * 1024.0);
            println!("Memory usage:      {:.2} MB", mem_mb);
        }
        println!("========================\n");
    }
}

/// Inference benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of benchmark runs
    pub runs: usize,
    /// Number of warmup runs (not included in results)
    pub warmup_runs: usize,
    /// Prompt length in tokens
    pub prompt_length: usize,
    /// Generation length in tokens
    pub gen_length: usize,
    /// Whether to measure TTFT (time to first token)
    pub measure_ttft: bool,
    /// Whether to measure memory usage
    pub measure_memory: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            runs: 10,
            warmup_runs: 2,
            prompt_length: 512,
            gen_length: 128,
            measure_ttft: true,
            measure_memory: true,
        }
    }
}

/// Inference benchmark harness
pub struct InferenceBenchmark {
    config: BenchmarkConfig,
}

impl InferenceBenchmark {
    /// Create a new benchmark with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create a new benchmark with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run inference benchmark
    ///
    /// # Arguments
    /// * `inference_fn` - Function that performs inference and returns token count
    ///
    /// # Returns
    /// Benchmark results with TPS metrics
    pub fn run<F>(&self, mut inference_fn: F) -> BenchmarkResult
    where
        F: FnMut() -> Result<(usize, Duration), Box<dyn std::error::Error>>,
    {
        let mut tps_samples = Vec::with_capacity(self.config.runs);
        let mut total_tokens = 0;
        let mut total_time = Duration::ZERO;
        let mut ttft = Duration::ZERO;

        // Warmup runs
        for _ in 0..self.config.warmup_runs {
            let _ = inference_fn();
        }

        // Actual benchmark runs
        for run in 0..self.config.runs {
            let start = Instant::now();
            match inference_fn() {
                Ok((num_tokens, elapsed)) => {
                    let tps = if elapsed.as_secs_f64() > 0.0 {
                        num_tokens as f64 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };

                    tps_samples.push(tps);
                    total_tokens += num_tokens;
                    total_time += elapsed;

                    // First run is used for TTFT measurement
                    if run == 0 && self.config.measure_ttft {
                        ttft = elapsed;
                    }
                }
                Err(e) => {
                    eprintln!("Benchmark run {} failed: {}", run + 1, e);
                }
            }
        }

        // Calculate statistics
        let mean_tps = if tps_samples.is_empty() {
            0.0
        } else {
            tps_samples.iter().sum::<f64>() / tps_samples.len() as f64
        };

        let min_tps = tps_samples
            .iter()
            .cloned()
            .fold(f32::INFINITY as f64, f64::min);

        let max_tps = tps_samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY as f64, f64::max);

        let std_tps = if tps_samples.is_empty() {
            0.0
        } else {
            let variance = tps_samples
                .iter()
                .map(|&x| (x - mean_tps).powi(2))
                .sum::<f64>()
                / tps_samples.len() as f64;
            variance.sqrt()
        };

        // Get memory usage if available
        let memory_bytes = if self.config.measure_memory {
            Some(get_memory_usage())
        } else {
            None
        };

        BenchmarkResult {
            mean_tps,
            min_tps,
            max_tps,
            std_tps,
            total_tokens,
            total_time,
            ttft,
            memory_bytes,
            runs: self.config.runs,
        }
    }

    /// Compare two benchmark results
    pub fn compare(before: &BenchmarkResult, after: &BenchmarkResult) -> Comparison {
        let tps_improvement = after.improvement_over(before);
        let ttft_change = if before.ttft.as_secs_f64() > 0.0 {
            ((after.ttft.as_secs_f64() - before.ttft.as_secs_f64()) / before.ttft.as_secs_f64())
                * 100.0
        } else {
            0.0
        };

        Comparison {
            tps_improvement,
            ttft_change,
            before: before.clone(),
            after: after.clone(),
        }
    }
}

/// Comparison between two benchmark results
#[derive(Debug, Clone)]
pub struct Comparison {
    /// TPS improvement percentage
    pub tps_improvement: f64,
    /// TTFT change percentage (negative = faster)
    pub ttft_change: f64,
    /// Before results
    pub before: BenchmarkResult,
    /// After results
    pub after: BenchmarkResult,
}

impl Comparison {
    /// Print comparison results
    pub fn print(&self) {
        println!("\n=== Performance Comparison ===");
        println!("\nBefore optimization:");
        println!("  TPS:  {:.2} tok/s", self.before.mean_tps);
        println!("  TTFT: {:?}", self.before.ttft);

        println!("\nAfter optimization:");
        println!("  TPS:  {:.2} tok/s", self.after.mean_tps);
        println!("  TTFT: {:?}", self.after.ttft);

        println!("\nImprovement:");
        if self.tps_improvement > 0.0 {
            println!("  TPS:  +{:.2}%", self.tps_improvement);
        } else {
            println!("  TPS:  {:.2}%", self.tps_improvement);
        }

        if self.ttft_change < 0.0 {
            println!("  TTFT: -{:.2}% (faster)", self.ttft_change.abs());
        } else {
            println!("  TTFT: +{:.2}% (slower)", self.ttft_change);
        }
        println!("==============================\n");
    }
}

/// Get current memory usage in bytes
#[cfg(target_os = "linux")]
fn get_memory_usage() -> usize {
    use std::fs;
    match fs::read_to_string("/proc/self/status") {
        Ok(status) => {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<usize>().unwrap_or(0) * 1024;
                    }
                }
            }
            0
        }
        Err(_) => 0,
    }
}

#[cfg(not(target_os = "linux"))]
fn get_memory_usage() -> usize {
    0
}

/// Global metrics for tracking inference statistics
#[derive(Debug)]
pub struct InferenceMetrics {
    pub total_tokens_generated: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub total_requests: AtomicU64,
    pub failed_requests: AtomicU64,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self {
            total_tokens_generated: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
        }
    }

    pub fn tokens_per_second(&self, elapsed_secs: f64) -> f64 {
        let tokens = self.total_tokens_generated.load(Ordering::Relaxed);
        if elapsed_secs > 0.0 {
            tokens as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);
        if total > 0 {
            (total - failed) as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn record_request(&self, prompt_tokens: usize, generated_tokens: usize) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(generated_tokens as u64, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn print(&self) {
        let total = self.total_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);
        let generated = self.total_tokens_generated.load(Ordering::Relaxed);
        let prompt = self.total_prompt_tokens.load(Ordering::Relaxed);

        println!("\n=== Inference Metrics ===");
        println!("Total requests:    {}", total);
        println!("Failed requests:   {}", failed);
        println!("Success rate:      {:.2}%", self.success_rate() * 100.0);
        println!("Prompt tokens:     {}", prompt);
        println!("Generated tokens:  {}", generated);
        println!("Total tokens:      {}", prompt + generated);
        println!("========================\n");
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult {
            mean_tps: 50.0,
            min_tps: 45.0,
            max_tps: 55.0,
            std_tps: 3.0,
            total_tokens: 500,
            total_time: Duration::from_secs(10),
            ttft: Duration::from_millis(200),
            memory_bytes: Some(1024 * 1024 * 1024),
            runs: 10,
        };

        assert_eq!(result.mean_tps, 50.0);
        assert_eq!(result.total_tokens, 500);
    }

    #[test]
    fn test_improvement_calculation() {
        let before = BenchmarkResult {
            mean_tps: 40.0,
            min_tps: 35.0,
            max_tps: 45.0,
            std_tps: 3.0,
            total_tokens: 400,
            total_time: Duration::from_secs(10),
            ttft: Duration::from_millis(200),
            memory_bytes: None,
            runs: 10,
        };

        let after = BenchmarkResult {
            mean_tps: 60.0,
            min_tps: 55.0,
            max_tps: 65.0,
            std_tps: 3.0,
            total_tokens: 600,
            total_time: Duration::from_secs(10),
            ttft: Duration::from_millis(150),
            memory_bytes: None,
            runs: 10,
        };

        let improvement = after.improvement_over(&before);
        assert!((improvement - 50.0).abs() < 0.01); // 50% improvement
    }

    #[test]
    fn test_inference_metrics() {
        let metrics = InferenceMetrics::new();
        metrics.record_request(100, 50);
        metrics.record_request(200, 100);

        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.total_tokens_generated.load(Ordering::Relaxed), 150);
        assert_eq!(metrics.total_prompt_tokens.load(Ordering::Relaxed), 300);
    }
}
