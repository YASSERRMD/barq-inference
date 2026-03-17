//! Advanced research features for Barq inference engine
//!
//! This module implements cutting-edge inference optimizations:

pub mod speculative;
pub mod flash_attention;
pub mod paged_attention;
pub mod prompt_cache;
pub mod rope_scaling;
pub mod moe;
pub mod tensor_parallel;
pub mod metrics;
pub mod metrics_server;
pub mod uds_server;
pub mod continuous_batching;
pub mod logging;

pub use speculative::SpeculativeDecoding;
pub use flash_attention::FlashAttention;
pub use paged_attention::PagedAttention;
pub use prompt_cache::PromptCache;
pub use rope_scaling::{RopeScaling, YaRNScaling};
pub use moe::MoEInference;
pub use tensor_parallel::TensorParallel;
