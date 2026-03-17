//! HTTP metrics server endpoint
//!
//! Exposes metrics and health checks via HTTP for monitoring systems.
//! Integrates with InferenceMetrics for real-time observability.

use std::sync::Arc;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode, Method};
use hyper::body::Bytes;
use http_body_util::Full;

use crate::metrics::{InferenceMetrics, MetricsHandle, MetricsResponse, HealthCheck};
use crate::logging::Logger;

/// Metrics server configuration
#[derive(Debug, Clone)]
pub struct MetricsServerConfig {
    /// Bind address
    pub bind_addr: SocketAddr,
    /// Metrics endpoint path
    pub metrics_path: String,
    /// Health check endpoint path
    pub health_path: String,
}

impl Default for MetricsServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9090".parse().unwrap(),
            metrics_path: "/metrics".to_string(),
            health_path: "/health".to_string(),
        }
    }
}

/// Metrics HTTP server
pub struct MetricsServer {
    config: MetricsServerConfig,
    metrics: MetricsHandle,
    logger: Arc<Logger>,
    start_time: std::time::Instant,
}

impl MetricsServer {
    pub fn new(config: MetricsServerConfig, metrics: MetricsHandle, logger: Arc<Logger>) -> Self {
        Self {
            config,
            metrics,
            logger,
            start_time: std::time::Instant::now(),
        }
    }

    /// Start the metrics server
    pub async fn start(self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(self.config.bind_addr).await?;
        self.logger.info(&format!(
            "Metrics server listening on http://{}",
            self.config.bind_addr
        ));

        let metrics = self.metrics;
        let logger = self.logger.clone();
        let metrics_path = self.config.metrics_path.clone();
        let health_path = self.config.health_path.clone();
        let start_time = self.start_time;

        loop {
            let (stream, _) = listener.accept().await?;

            let metrics = metrics.clone();
            let logger = logger.clone();
            let metrics_path = metrics_path.clone();
            let health_path = health_path.clone();
            let start_time = start_time;

            tokio::task::spawn(async move {
                let http = http1::Builder::new();
                let service = service_fn(move |req| {
                    Self::handle_request(
                        req,
                        metrics.clone(),
                        logger.clone(),
                        metrics_path.clone(),
                        health_path.clone(),
                        start_time,
                    )
                });

                let _ = http.serve_connection(stream, service).await;
            });
        }
    }

    /// Handle incoming HTTP request
    async fn handle_request(
        req: Request<Body>,
        metrics: MetricsHandle,
        logger: Arc<Logger>,
        metrics_path: String,
        health_path: String,
        start_time: std::time::Instant,
    ) -> Result<Response<Body>, hyper::Error> {
        let path = req.uri().path();
        let method = req.method();

        match (method, path) {
            (&Method::GET, p) if p == metrics_path => {
                logger.debug("Metrics endpoint called");
                let metrics_response = metrics.snapshot();
                let json = match serde_json::to_string_pretty(&metrics_response) {
                    Ok(j) => j,
                    Err(e) => {
                        return Ok(Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(Body::from(format!("Serialization error: {}", e)))
                            .unwrap());
                    }
                };

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "application/json")
                    .body(Body::from(json))
                    .unwrap())
            }

            (&Method::GET, p) if p == health_path => {
                logger.debug("Health check endpoint called");

                let uptime_secs = start_time.elapsed().as_secs_f64();
                let health = HealthCheck {
                    status: "healthy".to_string(),
                    uptime_seconds: uptime_secs,
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    metrics: serde_json::to_value(metrics.snapshot()).unwrap_or_default(),
                };

                let json = match serde_json::to_string_pretty(&health) {
                    Ok(j) => j,
                    Err(e) => {
                        return Ok(Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(Body::from(format!("Serialization error: {}", e)))
                            .unwrap());
                    }
                };

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "application/json")
                    .body(Body::from(json))
                    .unwrap())
            }

            (&Method::GET, "/") => {
                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "text/plain")
                    .body(Body::from("BarQ Inference Metrics Server\nEndpoints:\n  GET /metrics\n  GET /health\n"))
                    .unwrap())
            }

            _ => {
                Ok(Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::from("Not Found"))
                    .unwrap())
            }
        }
    }
}

/// Prometheus metrics exporter
pub struct PrometheusExporter {
    metrics: MetricsHandle,
}

impl PrometheusExporter {
    pub fn new(metrics: MetricsHandle) -> Self {
        Self { metrics }
    }

    /// Export metrics in Prometheus text format
    pub fn export_prometheus(&self) -> String {
        let snapshot = self.metrics.snapshot();

        let mut output = String::new();

        // Request metrics
        output.push_str("# HELP barq_requests_total Total number of requests\n");
        output.push_str("# TYPE barq_requests_total counter\n");
        output.push_str(&format!("barq_requests_total {}\n", snapshot.requests.total));

        output.push_str("\n# HELP barq_requests_successful Total successful requests\n");
        output.push_str("# TYPE barq_requests_successful counter\n");
        output.push_str(&format!("barq_requests_successful {}\n", snapshot.requests.successful));

        output.push_str("\n# HELP barq_requests_failed Total failed requests\n");
        output.push_str("# TYPE barq_requests_failed counter\n");
        output.push_str(&format!("barq_requests_failed {}\n", snapshot.requests.failed));

        output.push_str("\n# HELP barq_requests_active Currently active requests\n");
        output.push_str("# TYPE barq_requests_active gauge\n");
        output.push_str(&format!("barq_requests_active {}\n", snapshot.requests.active));

        output.push_str("\n# HELP barq_requests_peak_concurrent Peak concurrent requests\n");
        output.push_str("# TYPE barq_requests_peak_concurrent gauge\n");
        output.push_str(&format!("barq_requests_peak_concurrent {}\n", snapshot.requests.peak_concurrent));

        // Performance metrics
        output.push_str("\n# HELP barq_tokens_generated_total Total tokens generated\n");
        output.push_str("# TYPE barq_tokens_generated_total counter\n");
        output.push_str(&format!("barq_tokens_generated_total {}\n", snapshot.performance.total_tokens));

        output.push_str("\n# HELP barq_prompt_tokens_total Total prompt tokens processed\n");
        output.push_str("# TYPE barq_prompt_tokens_total counter\n");
        output.push_str(&format!("barq_prompt_tokens_total {}\n", snapshot.performance.prompt_tokens));

        output.push_str("\n# HELP barq_avg_tokens_per_second Average tokens per second\n");
        output.push_str("# TYPE barq_avg_tokens_per_second gauge\n");
        output.push_str(&format!("barq_avg_tokens_per_second {:.2}\n", snapshot.performance.avg_tps));

        output.push_str("\n# HELP barq_inference_time_ms_total Total inference time in milliseconds\n");
        output.push_str("# TYPE barq_inference_time_ms_total counter\n");
        output.push_str(&format!("barq_inference_time_ms_total {}\n", snapshot.performance.total_time_ms));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_server_config_default() {
        let config = MetricsServerConfig::default();
        assert_eq!(config.bind_addr.port(), 9090);
        assert_eq!(config.metrics_path, "/metrics");
        assert_eq!(config.health_path, "/health");
    }

    #[test]
    fn test_prometheus_exporter() {
        let metrics = Arc::new(InferenceMetrics::new());
        metrics.record_success(100, 200, 5000);

        let exporter = PrometheusExporter::new(metrics);
        let output = exporter.export_prometheus();

        assert!(output.contains("barq_requests_total"));
        assert!(output.contains("barq_tokens_generated_total"));
        assert!(output.contains("HELP"));
        assert!(output.contains("TYPE"));
    }
}
