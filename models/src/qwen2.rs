//! Qwen2 model implementation
//!
//! Qwen2 is the second generation of Qwen models
//! Features:
//! - Grouped Query Attention (GQA) support
//! - Improved RoPE implementation
//! - Better long-context handling
//! - SwiGLU activation

use crate::arch::LlmArch;
use crate::context::{ContextParams, ModelContext};
use crate::loader::Model;
use barq_core::error::{Error, Result};
use std::sync::Arc;

/// Qwen2 model specific implementations
pub struct Qwen2Model {
    model: Arc<Model>,
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

impl Qwen2Model {
    /// Create a new Qwen2 model wrapper
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Qwen2 {
            return Err(Error::Unsupported(format!(
                "Expected Qwen2 architecture, got {:?}",
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

    /// Check if this Qwen2 model uses GQA (Grouped Query Attention)
    pub fn has_gqa(&self) -> bool {
        // GQA is enabled when n_head_kv < n_head
        self.model.hparams.n_head_kv < self.model.hparams.n_head
    }

    /// Get the number of key-value heads for GQA
    pub fn n_kv_heads(&self) -> u32 {
        self.model.hparams.n_head_kv
    }

    /// Check if this Qwen2 model uses NTK-aware RoPE scaling
    pub fn has_ntk_scaling(&self) -> bool {
        self.model.hparams.rope_scaling_type == 1
    }
}

/// Qwen2MoE model (Mixture of Experts variant)
pub struct Qwen2MoEModel {
    model: Arc<Model>,
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

impl Qwen2MoEModel {
    /// Create a new Qwen2MoE model wrapper
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Qwen2Moe {
            return Err(Error::Unsupported(format!(
                "Expected Qwen2MoE architecture, got {:?}",
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

    /// Check if this Qwen2MoE model uses GQA
    pub fn has_gqa(&self) -> bool {
        self.model.hparams.n_head_kv < self.model.hparams.n_head
    }

    /// Get number of experts in the MoE model
    pub fn n_experts(&self) -> Option<u32> {
        // This would be extracted from GGUF metadata
        self.model
            .get_metadata("qwen2.n_experts")
            .and_then(|s| s.parse().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen2_creation() {
        // Placeholder - would need actual model data
    }
}
