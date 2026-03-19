//! LLaMA model implementation

use crate::arch::LlmArch;
use crate::context::{ContextParams, ModelContext};
use crate::loader::Model;
use barq_core::error::{Error, Result};
use std::sync::Arc;

/// LLaMA model specific implementations
pub struct LlamaModel {
    model: Arc<Model>,
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

impl LlamaModel {
    /// Create a new LLaMA model wrapper
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Llama {
            return Err(Error::Unsupported(format!(
                "Expected LLaMA architecture, got {:?}",
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
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_llama_creation() {
        // Would need actual model data to test
        // This is a placeholder
    }
}
