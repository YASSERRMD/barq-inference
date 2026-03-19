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

//! Model architecture implementations
//!
//! Support for 100+ LLM architectures including LLaMA, Mistral, Mixtral,
//! Qwen, GPT-2, BERT, and many more.

pub mod arch;
pub mod arch_registry;
pub mod context;
pub mod deepseek;
pub mod ffn;
pub mod kv_cache;
pub mod llama;
pub mod llava;
pub mod loader;
pub mod mistral;
pub mod mixtral;
pub mod moe_fused;
pub mod qwen;
pub mod qwen2;
pub mod qwen2vl;
pub mod qwen3;
pub mod transformer;
pub mod vision;
pub mod weight_cache;

#[cfg(test)]
pub(crate) mod test_support;

pub use arch::{LlmArch, LlmType};
pub use arch_registry::{ArchitectureRegistry, LlmArchTrait};
pub use context::ModelContext;
pub use deepseek::{DeepSeekMoEModel, DeepSeekModel};
pub use kv_cache::{AdvancedKVCache, KVCacheQuantization, KVCacheStats};
pub use llava::LlavaModel;
pub use loader::ModelLoader;
pub use moe_fused::{ExpertBatch, MoEFusedConfig, MoEFusedOps};
pub use qwen::QwenModel;
pub use qwen2::{Qwen2MoEModel, Qwen2Model};
pub use qwen2vl::Qwen2VlModel;
pub use qwen3::Qwen3Model;
pub use transformer::LlamaTransformer;
pub use vision::{ClipVisionEncoder, ImageInput, ImagePreprocessor, VisionEncoder};
pub use weight_cache::WeightCache;
