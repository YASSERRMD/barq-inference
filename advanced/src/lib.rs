//! Advanced research features for Barq inference engine
//!
//! This module implements cutting-edge inference optimizations:

pub mod speculative;
pub mod flash_attention;
pub mod paged_attention;
pub mod rope_scaling;
pub mod moe;
pub mod tensor_parallel;

pub use speculative::SpeculativeDecoding;
pub use flash_attention::FlashAttention;
pub use paged_attention::PagedAttention;
pub use rope_scaling::{RopeScaling, YaRNScaling};
pub use moe::MoEInference;
pub use tensor_parallel::TensorParallel;
