//! Mixtral (Mixture of Experts) model implementation

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
}

impl MixtralModel {
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

        Ok(Self {
            model,
            n_expert,
            n_expert_per_token,
        })
    }

    pub fn create_context(&self, params: ContextParams) -> Result<ModelContext> {
        ModelContext::new(Arc::clone(&self.model), params)
    }

    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    pub fn n_expert_per_token(&self) -> usize {
        self.n_expert_per_token
    }
}
