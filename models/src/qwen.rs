//! Qwen model implementation
//!
//! Qwen (通义千问) is a large language model by Alibaba Cloud
//! Features:
//! - NTK-aware RoPE scaling for extended context
//! - Specialized attention mechanism
//! - Dual-chunk attention for some variants

use crate::arch::LlmArch;
use crate::context::{ContextParams, ModelContext};
use crate::loader::Model;
use barq_core::error::{Error, Result};
use std::sync::Arc;

/// Qwen model specific implementations
pub struct QwenModel {
    model: Arc<Model>,
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

impl QwenModel {
    /// Create a new Qwen model wrapper
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Qwen {
            return Err(Error::Unsupported(format!(
                "Expected Qwen architecture, got {:?}",
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

    /// Check if this Qwen model uses NTK-aware RoPE scaling
    pub fn has_ntk_scaling(&self) -> bool {
        // Qwen models use NTK-aware scaling by default
        self.model.hparams.rope_scaling_type == 1
    }

    /// Get Qwen-specific RoPE frequency base
    pub fn rope_base(&self) -> f32 {
        self.model.hparams.rope_freq_base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_creation() {
        // Placeholder - would need actual model data
    }
}
