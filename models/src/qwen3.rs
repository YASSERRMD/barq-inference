//! Qwen3 model implementation
//!
//! Qwen3 is the third-generation Qwen family with:
//! - Grouped Query Attention (GQA)
//! - NTK-aware RoPE scaling for long context
//! - MoE variants handled separately in the registry

use crate::arch::LlmArch;
use crate::context::{ContextParams, ModelContext};
use crate::loader::Model;
use barq_core::error::{Error, Result};
use std::sync::Arc;

/// Qwen3 model specific implementations
pub struct Qwen3Model {
    model: Arc<Model>,
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

impl Qwen3Model {
    /// Create a new Qwen3 model wrapper
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Qwen3 {
            return Err(Error::Unsupported(format!(
                "Expected Qwen3 architecture, got {:?}",
                model.arch()
            )));
        }

        let transformer = Arc::new(crate::transformer::LlamaTransformer::new(model.clone())?);

        Ok(Self { model, transformer })
    }

    /// Create an inference context
    pub fn create_context(&self, params: ContextParams) -> Result<ModelContext> {
        ModelContext::new(
            Arc::clone(&self.model),
            params,
            Arc::clone(&self.transformer),
        )
    }

    /// Returns the model
    pub fn inner(&self) -> &Model {
        &self.model
    }

    /// Check if this Qwen3 model uses GQA (Grouped Query Attention)
    pub fn has_gqa(&self) -> bool {
        self.model.hparams.n_head_kv < self.model.hparams.n_head
    }

    /// Get the number of key-value heads for GQA
    pub fn n_kv_heads(&self) -> u32 {
        self.model.hparams.n_head_kv
    }

    /// Check if this Qwen3 model uses NTK-aware RoPE scaling
    pub fn has_ntk_scaling(&self) -> bool {
        self.model.hparams.rope_scaling_type == 1
    }

    /// Get Qwen3-specific RoPE frequency base
    pub fn rope_base(&self) -> f32 {
        self.model.hparams.rope_freq_base
    }

    /// Get Qwen3-specific RoPE scaling type
    pub fn rope_scaling_type(&self) -> u32 {
        self.model.hparams.rope_scaling_type
    }
}
