//! Mistral model implementation

use crate::arch::LlmArch;
use crate::context::{ContextParams, ModelContext};
use crate::loader::Model;
use barq_core::error::{Error, Result};
use std::sync::Arc;

/// Mistral model specific implementations
pub struct MistralModel {
    model: Arc<Model>,
}

impl MistralModel {
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Mistral {
            return Err(Error::Unsupported(format!(
                "Expected Mistral architecture, got {:?}",
                model.arch()
            )));
        }

        Ok(Self { model })
    }

    pub fn create_context(&self, params: ContextParams) -> Result<ModelContext> {
        ModelContext::new(Arc::clone(&self.model), params)
    }
}
