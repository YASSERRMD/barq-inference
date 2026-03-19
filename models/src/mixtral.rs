//! Mixtral (Mixture of Experts) model implementation
//!
//! Mixtral 8x7B and Mixtral 8x22B with:
//! - Sparse Mixture of Experts (8 experts, 2 active)
//! - Grouped Query Attention (GQA)
//! - Sliding Window Attention (SWA)
//! - RoPE scaling
//! - Expert load balancing

use crate::arch::LlmArch;
use crate::context::{ContextParams, ModelContext};
use crate::loader::Model;
use barq_core::error::{Error, Result};
use std::sync::Arc;

/// Mixtral MoE model implementation
pub struct MixtralModel {
    model: Arc<Model>,
    /// Number of experts
    n_expert: usize,
    /// Experts per token
    n_expert_per_token: usize,
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

impl MixtralModel {
    /// Create a new Mixtral model wrapper
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Mixtral {
            return Err(Error::Unsupported(format!(
                "Expected Mixtral architecture, got {:?}",
                model.arch()
            )));
        }

        // Extract MoE parameters from model metadata
        let n_expert = model
            .get_metadata("llama.expert_count")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8);

        let n_expert_per_token = model
            .get_metadata("llama.expert_used_count")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);

        let transformer = Arc::new(crate::transformer::LlamaTransformer::new(model.clone())?);

        Ok(Self {
            model,
            n_expert,
            n_expert_per_token,
            transformer,
        })
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

    /// Get number of experts in the MoE model
    /// Mixtral 8x7B has 8 experts, Mixtral 8x22B has 8 experts
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// Get number of active experts per token
    /// Mixtral uses top-2 expert selection
    pub fn n_expert_per_token(&self) -> usize {
        self.n_expert_per_token
    }

    /// Check if this Mixtral model uses GQA (Grouped Query Attention)
    /// Mixtral has 32 heads but only 8 KV heads
    pub fn has_gqa(&self) -> bool {
        self.model.hparams.n_head_kv < self.model.hparams.n_head
    }

    /// Get the number of key-value heads for GQA
    pub fn n_kv_heads(&self) -> u32 {
        self.model.hparams.n_head_kv
    }

    /// Check if model uses sliding window attention
    /// Mixtral uses a sliding window of 4096 tokens
    pub fn has_sliding_window(&self) -> bool {
        self.model
            .get_metadata("mixtral.sliding_window")
            .or_else(|| self.model.get_metadata("mistral.sliding_window"))
            .and_then(|s| s.parse::<u32>().ok())
            .is_some()
    }

    /// Get sliding window size in tokens
    pub fn sliding_window(&self) -> u32 {
        self.model
            .get_metadata("mixtral.sliding_window")
            .or_else(|| self.model.get_metadata("mistral.sliding_window"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096)
    }

    /// Check if load balancing loss is enabled
    /// Load balancing ensures even expert utilization
    pub fn has_load_balancing(&self) -> bool {
        self.model
            .get_metadata("mixtral.load_balancing")
            .and_then(|s| s.parse::<bool>().ok())
            .unwrap_or(true)
    }

    /// Get load balancing loss coefficient
    /// Controls the strength of expert load balancing
    pub fn load_balancing_coeff(&self) -> f32 {
        self.model
            .get_metadata("mixtral.load_balancing_coeff")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.01)
    }

    /// Check if this is Mixtral 8x7B
    pub fn is_8x7b(&self) -> bool {
        self.model.hparams.n_ff == 14336
    }

    /// Check if this is Mixtral 8x22B
    pub fn is_8x22b(&self) -> bool {
        self.model.hparams.n_ff == 45056
    }

    /// Check if model uses RoPE scaling
    pub fn has_rope_scaling(&self) -> bool {
        self.model.hparams.rope_scaling_type > 0
    }

    /// Get RoPE scaling type
    /// 0 = none, 1 = linear, 2 = yarn
    pub fn rope_scaling_type(&self) -> u32 {
        self.model.hparams.rope_scaling_type
    }

    /// Get expert routing type
    /// "topk" = top-k routing, "affine" = affine routing
    pub fn routing_type(&self) -> String {
        self.model
            .get_metadata("mixtral.routing_type")
            .cloned()
            .unwrap_or_else(|| "topk".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_test_gguf_file;
    use barq_core::gguf::GgufValue;
    use std::sync::Arc;

    async fn load_mixtral_model() -> Arc<Model> {
        let path = write_test_gguf_file(
            "mixtral",
            &[
                (
                    "general.architecture",
                    GgufValue::String("mixtral".to_string()),
                ),
                ("llama.block_count", GgufValue::Uint32(32)),
                ("llama.attention.head_count", GgufValue::Uint32(32)),
                ("llama.attention.head_count_kv", GgufValue::Uint32(8)),
                ("llama.embedding_length", GgufValue::Uint32(4096)),
                ("llama.feed_forward_length", GgufValue::Uint32(14_336)),
                ("llama.context_length", GgufValue::Uint32(32_768)),
                ("llama.rope.scaling.type", GgufValue::Uint32(1)),
                ("llama.expert_count", GgufValue::Uint32(8)),
                ("llama.expert_used_count", GgufValue::Uint32(2)),
                ("mixtral.sliding_window", GgufValue::Uint32(4_096)),
                ("mixtral.load_balancing", GgufValue::Bool(false)),
                ("mixtral.load_balancing_coeff", GgufValue::Float32(0.02)),
                (
                    "mixtral.routing_type",
                    GgufValue::String("topk".to_string()),
                ),
            ],
        );
        Arc::new(Model::load(&path).await.unwrap())
    }

    #[tokio::test]
    async fn test_mixtral_creation() {
        let model = load_mixtral_model().await;
        let wrapper = MixtralModel::new(model).unwrap();

        assert_eq!(wrapper.n_expert(), 8);
        assert_eq!(wrapper.n_expert_per_token(), 2);
        assert!(wrapper.has_gqa());
        assert_eq!(wrapper.n_kv_heads(), 8);
        assert!(wrapper.has_sliding_window());
        assert_eq!(wrapper.sliding_window(), 4_096);
        assert!(!wrapper.has_load_balancing());
        assert!((wrapper.load_balancing_coeff() - 0.02).abs() < f32::EPSILON);
        assert!(wrapper.is_8x7b());
        assert!(!wrapper.is_8x22b());
        assert!(wrapper.has_rope_scaling());
        assert_eq!(wrapper.rope_scaling_type(), 1);
        assert_eq!(wrapper.routing_type(), "topk");
        assert!(wrapper.create_context(ContextParams::default()).is_ok());
    }
}
