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
pub mod context;
pub mod deepseek;
pub mod ffn;
pub mod kv_cache;
pub mod llama;
pub mod loader;
pub mod mistral;
pub mod mixtral;
pub mod qwen;
pub mod qwen2;
pub mod transformer;

pub use arch::{LlmArch, LlmType};
pub use context::ModelContext;
pub use deepseek::{DeepSeekMoEModel, DeepSeekModel};
pub use kv_cache::{AdvancedKVCache, KVCacheStats};
pub use loader::ModelLoader;
pub use qwen::QwenModel;
pub use qwen2::{Qwen2MoEModel, Qwen2Model};
pub use transformer::LlamaTransformer;
