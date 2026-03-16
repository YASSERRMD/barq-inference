//! Observability and production metrics
//!
//! Provides metrics exposure, logging integration, and graceful
//! context reset for production deployments.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;

/// Global inference metrics
#[derive(Debug, Serialize)]
pub struct InferenceMetrics {
    /// Total requests processed
    pub total_requests: AtomicU64,
    /// Successful requests
    pub successful_requests: AtomicU64,
    /// Failed requests
    pub failed_requests: AtomicU64,
    /// Total tokens generated
    pub total_tokens_generated: AtomicU64,
    /// Total prompt tokens processed
    pub total_prompt_tokens: AtomicU64,
    /// Total inference time (milliseconds)
    pub total_inference_time_ms: AtomicU64,
    /// Current active requests
    pub active_requests: AtomicUsize,
    /// Peak concurrent requests
    pub peak_concurrent_requests: AtomicUsize,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_inference_time_ms: AtomicU64::new(0),
            active_requests: AtomicUsize::new(0),
            peak_concurrent_requests: AtomicUsize::new(0),
        }
    }

    pub fn record_request_start(&self) -> RequestGuard {
        let old_count = self.active_requests.fetch_add(1, Ordering::Relaxed);
        let mut peak = self.peak_concurrent_requests.load(Ordering::Relaxed);

        while old_count + 1 > peak {
            match self.peak_concurrent_requests.compare_exchange_weak(
                peak,
                old_count + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }

        self.total_requests.fetch_add(1, Ordering::Relaxed);

        RequestGuard {
            metrics: Arc::downgrade(&self.inner),
            start: Instant::now(),
        }
    }

    pub fn record_success(&self, prompt_tokens: usize, generated_tokens: usize, duration_ms: u64) {
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_tokens_generated.fetch_add(generated_tokens as u64, Ordering::Relaxed);
        self.total_prompt_tokens.fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.total_inference_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        let successful = self.successful_requests.load(Ordering::Relaxed);

        if total == 0 {
            0.0
        } else {
            successful as f64 / total as f64
        }
    }

    pub fn average_tokens_per_second(&self) -> f64 {
        let tokens = self.total_tokens_generated.load(Ordering::Relaxed);
        let time_ms = self.total_inference_time_ms.load(Ordering::Relaxed);

        if time_ms == 0 {
            0.0
        } else {
            (tokens as f64) / (time_ms as f64 / 1000.0)
        }
    }

    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// Inner wrapper for Arc downgrading in RequestGuard
struct InferenceMetricsInner {
    metrics: InferenceMetrics,
}

/// RAII guard for request tracking
pub struct RequestGuard {
    metrics: Option<std::sync::Weak<InferenceMetricsInner>>,
    start: Instant,
}

impl RequestGuard {
    pub fn complete(self, prompt_tokens: usize, generated_tokens: usize) {
        if let Some(inner) = self.metrics {
            if let Some(metrics) = inner.upgrade() {
                let duration = self.start.elapsed();
                metrics.record_success(prompt_tokens, generated_tokens, duration.as_millis() as u64);
            }
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        if let Some(inner) = &self.metrics {
            if let Some(metrics) = inner.upgrade() {
                metrics.active_requests.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

/// Health check endpoint data
#[derive(Debug, Serialize)]
pub struct HealthCheck {
    pub status: String,
    pub uptime_seconds: f64,
    pub version: String,
    pub metrics: serde_json::Value,
}

/// Metrics endpoint data
#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub timestamp: i64,
    pub uptime_seconds: f64,
    pub requests: MetricSnapshot,
    pub performance: PerformanceSnapshot,
}

#[derive(Debug, Serialize)]
pub struct MetricSnapshot {
    pub total: u64,
    pub successful: u64,
    pub failed: u64,
    pub success_rate: f64,
    pub active: usize,
    pub peak_concurrent: usize,
}

#[derive(Debug, Serialize)]
pub struct PerformanceSnapshot {
    pub total_tokens: u64,
    pub prompt_tokens: u64,
    pub avg_tps: f64,
    pub total_time_ms: u64,
}

impl InferenceMetrics {
    pub fn snapshot(&self) -> MetricsResponse {
        MetricsResponse {
            timestamp: chrono::Utc::now().timestamp(),
            uptime_seconds: get_process_uptime(),
            requests: MetricSnapshot {
                total: self.total_requests.load(Ordering::Relaxed),
                successful: self.successful_requests.load(Ordering::Relaxed),
                failed: self.failed_requests.load(Ordering::Relaxed),
                success_rate: self.success_rate(),
                active: self.active_requests.load(Ordering::Relaxed),
                peak_concurrent: self.peak_concurrent_requests.load(Ordering::Relaxed),
            },
            performance: PerformanceSnapshot {
                total_tokens: self.total_tokens_generated.load(Ordering::Relaxed),
                prompt_tokens: self.total_prompt_tokens.load(Ordering::Relaxed),
                avg_tps: self.average_tokens_per_second(),
                total_time_ms: self.total_inference_time_ms.load(Ordering::Relaxed),
            },
        }
    }
}

/// Get process uptime in seconds
fn get_process_uptime() -> f64 {
    use std::time::Instant;

    // This would need to be stored at application start
    // For now, return a placeholder
    0.0
}

/// Graceful context reset
pub fn check_context_health(ctx_tokens: usize, ctx_capacity: usize, threshold: f32) -> bool {
    if ctx_capacity == 0 {
        return true;
    }

    let usage = ctx_tokens as f64 / ctx_capacity as f64;

    if usage > threshold {
        return false; // Should reset
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = InferenceMetrics::new();
        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.success_rate(), 0.0);
    }

    #[test]
    fn test_metrics_recording() {
        let metrics = InferenceMetrics::new();

        metrics.record_success(10, 20, 1000);
        assert_eq!(metrics.successful_requests.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_tokens_generated.load(Ordering::Relaxed), 20);
        assert_eq!(metrics.total_prompt_tokens.load(Ordering::Relaxed), 10);
        assert_eq!(metrics.total_inference_time_ms.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_metrics_success_rate() {
        let metrics = InferenceMetrics::new();

        metrics.record_success(10, 20, 1000);
        metrics.record_success(10, 20, 1000);
        metrics.record_failure();

        assert!((metrics.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_metrics_avg_tps() {
        let metrics = InferenceMetrics::new();

        metrics.record_success(100, 100, 10000); // 100 tokens in 10 seconds
        assert!((metrics.average_tokens_per_second() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_context_health() {
        assert!(check_context_health(1000, 8192, 0.9)); // 12% usage
        assert!(!check_context_health(8000, 8192, 0.9)); // 97% usage
    }
}
