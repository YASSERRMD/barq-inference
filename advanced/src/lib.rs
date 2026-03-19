#![allow(
    clippy::all,
    unexpected_cfgs,
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code
)]
#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code,
    clippy::needless_update,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::let_and_return,
    clippy::manual_range_contains
)]

//! Advanced research features for Barq inference engine
//!
//! This module implements cutting-edge inference optimizations:

pub mod batch_engine;
pub mod benchmarks;
pub mod continuous_batching;
pub mod flash_attention;
pub mod logging;
pub mod metrics;
pub mod moe;
pub mod paged_attention;
pub mod prompt_cache;
pub mod rope_scaling;
pub mod speculative;
pub mod tensor_parallel;
pub mod uds_server;
// pub mod metrics_server;  // TODO: Fix hyper compatibility issues

pub use batch_engine::{BatchEngine, BatchEngineHandle, BatchRequest};
pub use benchmarks::{run_benchmark, BenchmarkResult};
pub use continuous_batching::{Batch, BatchScheduler, ContinuousBatchingConfig};
pub use flash_attention::FlashAttention;
pub use logging::{init_logger, logger, LogLevel, Logger, LoggingConfig, RequestLogger};
pub use metrics::{
    check_context_health, ContextHealth, ContextManager, HealthCheck, InferenceMetrics,
    MetricsHandle, MetricsResponse, RequestGuard,
};
pub use moe::MoEInference;
pub use paged_attention::PagedAttention;
pub use prompt_cache::PromptCache;
pub use rope_scaling::{RopeScaling, YaRNScaling};
pub use speculative::{MockSpeculativeDecoder, SpeculativeDecoding, SpeculativeEngine};
pub use tensor_parallel::TensorParallel;
pub use uds_server::{
    InferenceClient, InferenceRequest, InferenceResponse, InferenceServer, ServerConfig,
};
// pub use metrics_server::{MetricsServer, MetricsServerConfig, PrometheusExporter};
