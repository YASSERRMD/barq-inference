//! Multi-user benchmarking for UDS inference server
//!
//! Synchronizes multiple concurrent clients to measure:
//! - Aggregate throughput (tokens/sec)
//! - Latency distribution (P50, P90, P99)
//! - Continuous batching efficiency

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Barrier;

use crate::uds_server::{InferenceClient, InferenceRequest};
use barq_core::error::Result;

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub total_requests: usize,
    pub total_tokens: usize,
    pub total_time: Duration,
    pub avg_tps: f64,
    pub p50_latency_ms: u64,
    pub p99_latency_ms: u64,
}

/// Run a multi-user benchmark
pub async fn run_benchmark(
    socket_path: PathBuf,
    num_users: usize,
    requests_per_user: usize,
    prompt: &str,
    max_tokens: usize,
) -> Result<BenchmarkResult> {
    let client = Arc::new(InferenceClient::new(socket_path));
    let barrier = Arc::new(Barrier::new(num_users));
    let mut handles = Vec::new();

    let start = Instant::now();

    for user_idx in 0..num_users {
        let client = client.clone();
        let barrier = barrier.clone();
        let prompt = prompt.to_string();

        let handle = tokio::spawn(async move {
            let mut latencies = Vec::new();
            let mut user_tokens = 0;

            // Wait for all users to be ready
            barrier.wait().await;

            for req_idx in 0..requests_per_user {
                let request = InferenceRequest {
                    id: format!("user-{}-req-{}", user_idx, req_idx),
                    prompt: prompt.clone(),
                    max_tokens,
                    temperature: 0.7,
                };

                let req_start = Instant::now();
                match client.infer(request).await {
                    Ok(resp) => {
                        latencies.push(req_start.elapsed().as_millis() as u64);
                        user_tokens += resp.tokens_generated;
                    }
                    Err(e) => eprintln!("User {} request {} failed: {}", user_idx, req_idx, e),
                }
            }
            (latencies, user_tokens)
        });
        handles.push(handle);
    }

    let mut all_latencies = Vec::new();
    let mut total_tokens = 0;

    for handle in handles {
        if let Ok((latencies, tokens)) = handle.await {
            all_latencies.extend(latencies);
            total_tokens += tokens;
        }
    }

    let total_time = start.elapsed();
    all_latencies.sort_unstable();

    let p50 = if !all_latencies.is_empty() {
        all_latencies[all_latencies.len() * 50 / 100]
    } else {
        0
    };

    let p99 = if !all_latencies.is_empty() {
        all_latencies[all_latencies.len() * 99 / 100]
    } else {
        0
    };

    Ok(BenchmarkResult {
        total_requests: num_users * requests_per_user,
        total_tokens,
        total_time,
        avg_tps: total_tokens as f64 / total_time.as_secs_f64(),
        p50_latency_ms: p50,
        p99_latency_ms: p99,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uds_server::{InferenceServer, ServerConfig};

    #[tokio::test]
    async fn test_benchmark_scaffold() {
        let socket_path = PathBuf::from("/tmp/test-bench.sock");
        let config = ServerConfig {
            socket_path: socket_path.clone(),
            max_concurrent: 4,
            queue_size: 10,
            worker_threads: 1,
        };

        let mut server = InferenceServer::new(config);

        // Start server in background
        let server_path = socket_path.clone();
        tokio::spawn(async move {
            let mut s = InferenceServer::new(ServerConfig {
                socket_path: server_path,
                ..Default::default()
            });
            let _ = s.start().await;
        });

        // Wait for server to start
        tokio::time::sleep(Duration::from_millis(200)).await;

        let result = run_benchmark(
            socket_path.clone(),
            2, // 2 users
            2, // 2 requests each
            "Test prompt",
            10,
        )
        .await;

        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.total_requests, 4);

        let _ = std::fs::remove_file(socket_path);
    }
}
