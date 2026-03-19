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
    use crate::test_support::write_test_gguf_file;
    use barq_core::gguf::GgufValue;
    use std::sync::Arc;

    async fn load_qwen_model() -> Arc<Model> {
        let path = write_test_gguf_file(
            "qwen",
            &[
                (
                    "general.architecture",
                    GgufValue::String("qwen".to_string()),
                ),
                ("llama.block_count", GgufValue::Uint32(24)),
                ("llama.attention.head_count", GgufValue::Uint32(32)),
                ("llama.attention.head_count_kv", GgufValue::Uint32(32)),
                ("llama.embedding_length", GgufValue::Uint32(4096)),
                ("llama.feed_forward_length", GgufValue::Uint32(11_008)),
                ("llama.context_length", GgufValue::Uint32(32_768)),
                ("qwen.rope.freq_base", GgufValue::Float32(1_000_000.0)),
                ("qwen.rope.scaling.type", GgufValue::Uint32(1)),
            ],
        );
        Arc::new(Model::load(&path).await.unwrap())
    }

    #[tokio::test]
    async fn test_qwen_creation() {
        let model = load_qwen_model().await;
        let wrapper = QwenModel::new(model).unwrap();

        assert!(wrapper.has_ntk_scaling());
        assert_eq!(wrapper.rope_base(), 1_000_000.0);
        assert!(wrapper.create_context(ContextParams::default()).is_ok());
    }
}
