//! Advanced research features for Barq inference engine
//!
//! This module implements cutting-edge inference optimizations:

pub mod speculative;
pub mod flash_attention;
pub mod paged_attention;
pub mod rope_scaling;
pub mod moe;
pub mod tensor_parallel;
pub mod uds_server;
pub mod continuous_batching;
pub mod metrics;
pub mod logging;
pub mod metrics_server;

pub use speculative::SpeculativeDecoding;
pub use flash_attention::FlashAttention;
pub use paged_attention::PagedAttention;
pub use rope_scaling::{RopeScaling, YaRNScaling};
pub use moe::MoEInference;
pub use tensor_parallel::TensorParallel;
pub use uds_server::{InferenceServer, InferenceClient, InferenceRequest, InferenceResponse, ServerConfig};
pub use continuous_batching::{BatchScheduler, Batch, ContinuousBatchingConfig};
pub use metrics::{InferenceMetrics, MetricsHandle, RequestGuard, HealthCheck, MetricsResponse, check_context_health, ContextManager, ContextHealth};
pub use logging::{Logger, RequestLogger, LogLevel, LoggingConfig, init_logger, logger};
pub use metrics_server::{MetricsServer, MetricsServerConfig, PrometheusExporter};
