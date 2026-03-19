//! DeepSeek model implementation
//!
//! DeepSeek is a family of efficient language models with:
//! - Multi-head Latent Attention (MLA) for efficient KV cache compression
//! - Mixture of Experts (MoE) for DeepSeek-MoE variants
//! - SwiGLU activation function
//! - Yarn RoPE scaling for extended context

use crate::arch::LlmArch;
use crate::context::{ContextParams, ModelContext};
use crate::loader::Model;
use barq_core::error::{Error, Result};
use std::sync::Arc;

/// DeepSeek model specific implementations
///
/// Features:
/// - MLA (Multi-head Latent Attention) with compressed KV cache
/// - Yarn RoPE scaling for long context
/// - SwiGLU activation
pub struct DeepSeekModel {
    model: Arc<Model>,
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

impl DeepSeekModel {
    /// Create a new DeepSeek model wrapper
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::DeepSeek {
            return Err(Error::Unsupported(format!(
                "Expected DeepSeek architecture, got {:?}",
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

    /// Check if this DeepSeek model uses MLA (Multi-head Latent Attention)
    pub fn has_mla(&self) -> bool {
        // All DeepSeek models use MLA
        true
    }

    /// Get MLA compression ratio
    /// MLA compresses KV cache by latent vectors
    pub fn mla_compression_ratio(&self) -> f32 {
        // Typical MLA compresses KV cache to 1/4 or 1/8 of original size
        // This can be extracted from GGUF metadata
        self.model
            .get_metadata("deepseek.mla_compression")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.25) // Default 4x compression
    }

    /// Get number of latent attention heads
    pub fn n_latent_heads(&self) -> u32 {
        self.model
            .get_metadata("deepseek.n_latent_heads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.model.hparams.n_head)
    }

    /// Check if model uses Yarn RoPE scaling
    pub fn has_yarn_scaling(&self) -> bool {
        self.model.hparams.rope_scaling_type == 2
    }

    /// Get SwiGLU intermediate size multiplier
    /// DeepSeek uses a modified SwiGLU with specific multiplier
    pub fn swiglu_multiplier(&self) -> f32 {
        self.model
            .get_metadata("deepseek.swiglu_multiplier")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.6667) // Default 8/3 multiplier
    }
}

/// DeepSeek-MoE model (Mixture of Experts variant)
///
/// Features:
/// - MLA for efficient attention
/// - MoE routing with expert selection
/// - Auxiliary loss load balancing
pub struct DeepSeekMoEModel {
    model: Arc<Model>,
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

impl DeepSeekMoEModel {
    /// Create a new DeepSeekMoE model wrapper
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::DeepSeekMoE {
            return Err(Error::Unsupported(format!(
                "Expected DeepSeekMoE architecture, got {:?}",
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

    /// Check if this DeepSeekMoE model uses MLA
    pub fn has_mla(&self) -> bool {
        true
    }

    /// Get number of experts in the MoE model
    pub fn n_experts(&self) -> u32 {
        // DeepSeek-MoE V2 has 160 experts, V3 has 64 experts
        self.model
            .get_metadata("deepseek.n_experts")
            .and_then(|s| s.parse().ok())
            .unwrap_or(160) // Default to DeepSeek-MoE V2
    }

    /// Get number of active experts per token
    pub fn n_active_experts(&self) -> u32 {
        self.model
            .get_metadata("deepseek.n_active_experts")
            .and_then(|s| s.parse().ok())
            .unwrap_or(6) // Default 6 active experts
    }

    /// Get MoE routing type
    pub fn routing_type(&self) -> String {
        self.model
            .get_metadata("deepseek.routing_type")
            .cloned()
            .unwrap_or_else(|| "affine".to_string())
    }

    /// Check if load balancing loss is enabled
    pub fn has_load_balancing(&self) -> bool {
        self.model
            .get_metadata("deepseek.load_balancing")
            .and_then(|s| s.parse().ok())
            .unwrap_or(true)
    }

    /// Get MLA compression ratio
    pub fn mla_compression_ratio(&self) -> f32 {
        self.model
            .get_metadata("deepseek.mla_compression")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.25)
    }

    /// Get number of latent attention heads
    pub fn n_latent_heads(&self) -> u32 {
        self.model
            .get_metadata("deepseek.n_latent_heads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.model.hparams.n_head)
    }

    /// Get SwiGLU intermediate size multiplier
    pub fn swiglu_multiplier(&self) -> f32 {
        self.model
            .get_metadata("deepseek.swiglu_multiplier")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.6667)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_test_gguf_file;
    use barq_core::gguf::GgufValue;
    use std::sync::Arc;

    async fn load_deepseek_model() -> Arc<Model> {
        let path = write_test_gguf_file(
            "deepseek",
            &[
                (
                    "general.architecture",
                    GgufValue::String("deepseek".to_string()),
                ),
                ("deepseek.block_count", GgufValue::Uint32(28)),
                ("deepseek.attention.head_count", GgufValue::Uint32(32)),
                ("deepseek.attention.head_count_kv", GgufValue::Uint32(8)),
                ("deepseek.embedding_length", GgufValue::Uint32(4096)),
                ("deepseek.intermediate_size", GgufValue::Uint32(11_008)),
                ("deepseek.context_length", GgufValue::Uint32(131_072)),
                ("deepseek.rope.freq_base", GgufValue::Float32(10_000.0)),
                ("deepseek.rope.freq_scale", GgufValue::Float32(1.0)),
                ("deepseek.rope.scaling.type", GgufValue::Uint32(2)),
                ("deepseek.mla_compression", GgufValue::Float32(0.25)),
                ("deepseek.n_latent_heads", GgufValue::Uint32(8)),
                ("deepseek.swiglu_multiplier", GgufValue::Float32(2.6667)),
            ],
        );
        Arc::new(Model::load(&path).await.unwrap())
    }

    async fn load_deepseek_moe_model() -> Arc<Model> {
        let path = write_test_gguf_file(
            "deepseek_moe",
            &[
                (
                    "general.architecture",
                    GgufValue::String("deepseek.moe".to_string()),
                ),
                ("deepseek.block_count", GgufValue::Uint32(30)),
                ("deepseek.attention.head_count", GgufValue::Uint32(48)),
                ("deepseek.attention.head_count_kv", GgufValue::Uint32(8)),
                ("deepseek.embedding_length", GgufValue::Uint32(6144)),
                ("deepseek.intermediate_size", GgufValue::Uint32(16_384)),
                ("deepseek.context_length", GgufValue::Uint32(131_072)),
                ("deepseek.rope.scaling.type", GgufValue::Uint32(2)),
                ("deepseek.mla_compression", GgufValue::Float32(0.125)),
                ("deepseek.n_latent_heads", GgufValue::Uint32(16)),
                ("deepseek.swiglu_multiplier", GgufValue::Float32(2.75)),
                ("deepseek.n_experts", GgufValue::Uint32(64)),
                ("deepseek.n_active_experts", GgufValue::Uint32(6)),
                (
                    "deepseek.routing_type",
                    GgufValue::String("affine".to_string()),
                ),
                ("deepseek.load_balancing", GgufValue::Bool(false)),
            ],
        );
        Arc::new(Model::load(&path).await.unwrap())
    }

    #[tokio::test]
    async fn test_deepseek_creation() {
        let model = load_deepseek_model().await;
        let wrapper = DeepSeekModel::new(model).unwrap();

        assert!(wrapper.has_mla());
        assert!(wrapper.has_yarn_scaling());
        assert!((wrapper.mla_compression_ratio() - 0.25).abs() < f32::EPSILON);
        assert_eq!(wrapper.n_latent_heads(), 8);
        assert!((wrapper.swiglu_multiplier() - 2.6667).abs() < 1e-6);
        assert!(wrapper.create_context(ContextParams::default()).is_ok());
    }

    #[tokio::test]
    async fn test_deepseek_moe_creation() {
        let model = load_deepseek_moe_model().await;
        let wrapper = DeepSeekMoEModel::new(model).unwrap();

        assert!(wrapper.has_mla());
        assert_eq!(wrapper.n_experts(), 64);
        assert_eq!(wrapper.n_active_experts(), 6);
        assert_eq!(wrapper.routing_type(), "affine");
        assert!(!wrapper.has_load_balancing());
        assert!((wrapper.mla_compression_ratio() - 0.125).abs() < f32::EPSILON);
        assert_eq!(wrapper.n_latent_heads(), 16);
        assert!((wrapper.swiglu_multiplier() - 2.75).abs() < 1e-6);
        assert!(wrapper.create_context(ContextParams::default()).is_ok());
    }
}
