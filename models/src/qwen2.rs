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
    use crate::test_support::write_test_gguf_file;
    use barq_core::gguf::GgufValue;
    use std::sync::Arc;

    async fn load_qwen2_model() -> Arc<Model> {
        let path = write_test_gguf_file(
            "qwen2",
            &[
                (
                    "general.architecture",
                    GgufValue::String("qwen2".to_string()),
                ),
                ("general.vocab_size", GgufValue::Uint32(151_936)),
                ("qwen.block_count", GgufValue::Uint32(28)),
                ("qwen.attention.head_count", GgufValue::Uint32(40)),
                ("qwen.attention.head_count_kv", GgufValue::Uint32(8)),
                ("qwen.embedding_length", GgufValue::Uint32(5120)),
                ("qwen.intermediate_size", GgufValue::Uint32(13_824)),
                ("qwen.context_length", GgufValue::Uint32(32_768)),
                ("qwen.rope.freq_base", GgufValue::Float32(1_000_000.0)),
                ("qwen.rope.scaling.type", GgufValue::Uint32(1)),
            ],
        );
        Arc::new(Model::load(&path).await.unwrap())
    }

    async fn load_qwen2_moe_model() -> Arc<Model> {
        let path = write_test_gguf_file(
            "qwen2_moe",
            &[
                (
                    "general.architecture",
                    GgufValue::String("qwen2.moe".to_string()),
                ),
                ("qwen2.n_experts", GgufValue::Uint32(4)),
                ("qwen.block_count", GgufValue::Uint32(28)),
                ("qwen.attention.head_count", GgufValue::Uint32(40)),
                ("qwen.attention.head_count_kv", GgufValue::Uint32(8)),
                ("qwen.embedding_length", GgufValue::Uint32(5120)),
                ("qwen.intermediate_size", GgufValue::Uint32(13_824)),
                ("qwen.context_length", GgufValue::Uint32(32_768)),
                ("qwen.rope.freq_base", GgufValue::Float32(1_000_000.0)),
                ("qwen.rope.scaling.type", GgufValue::Uint32(1)),
            ],
        );
        Arc::new(Model::load(&path).await.unwrap())
    }

    #[tokio::test]
    async fn test_qwen2_creation() {
        let model = load_qwen2_model().await;
        let wrapper = Qwen2Model::new(model).unwrap();

        assert!(wrapper.has_gqa());
        assert_eq!(wrapper.n_kv_heads(), 8);
        assert!(wrapper.has_ntk_scaling());
        assert!(wrapper.create_context(ContextParams::default()).is_ok());
    }

    #[tokio::test]
    async fn test_qwen2_moe_creation() {
        let model = load_qwen2_moe_model().await;
        let wrapper = Qwen2MoEModel::new(model).unwrap();

        assert!(wrapper.has_gqa());
        assert_eq!(wrapper.n_experts(), Some(4));
        assert!(wrapper.create_context(ContextParams::default()).is_ok());
    }
}
