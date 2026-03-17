//! Speculative decoding engine integration
//!
//! Uses a smaller draft model to predict multiple tokens ahead,
//! then verifies them with the main model for 1.5-3x speedup.
//!
//! Architecture:
//! ┌─────────────────────────────────────────────────┐
//! │  Rust Orchestration Layer                        │
//! │                                                   │
//! │  ┌─────────────┐    draft tokens    ┌──────────┐ │
//! │  │ Draft Model │ ─────────────────> │  Target  │ │
//! │  │ (1B-3B)    │ <── verify/reject ─ │  Model   │ │
//! │  │             │                    │ (7B-70B) │ │
//! │  └─────────────┘                    └──────────┘ │
//! └─────────────────────────────────────────────────┘

use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::Semaphore;

use crate::context::{ModelContext, ContextParams};
use crate::loader::Model;
use crate::kv_cache::AdvancedKVCache;
use barq_core::error::{Error, Result};

/// Speculative decoding configuration
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate ahead (sweet spot: 8-16)
    pub draft_max: usize,
    /// Minimum acceptance probability (0.0-1.0)
    pub draft_p_min: f32,
    /// Split probability for speculative tree
    pub draft_p_split: f32,
    /// Draft context size (can be smaller than target)
    pub ctx_size_draft: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_max: 16,           // Recommended sweet spot
            draft_p_min: 0.8,        // High acceptance threshold
            draft_p_split: 0.1,      // Conservative split
            ctx_size_draft: 4096,    // Smaller draft context
        }
    }
}

impl SpeculativeConfig {
    /// Optimized for code generation (deterministic tasks)
    pub fn code_generation() -> Self {
        Self {
            draft_max: 16,
            draft_p_min: 0.9,        // Higher threshold for code
            draft_p_split: 0.05,
            ctx_size_draft: 4096,
        }
    }

    /// Optimized for creative writing
    pub fn creative() -> Self {
        Self {
            draft_max: 8,             // More conservative
            draft_p_min: 0.7,        // Lower threshold for creativity
            draft_p_split: 0.15,
            ctx_size_draft: 4096,
        }
    }

    /// Maximum speed mode
    pub fn max_speed() -> Self {
        Self {
            draft_max: 16,
            draft_p_min: 0.5,        // More aggressive acceptance
            draft_p_split: 0.1,
            ctx_size_draft: 2048,    // Smaller context for speed
        }
    }
}

/// Speculative decoding engine with dual model setup
pub struct SpeculativeEngine {
    /// Main (target) model - larger, more accurate
    pub target_model: Arc<Model>,
    /// Draft model - smaller, faster
    pub draft_model: Arc<Model>,
    /// Target model context
    pub target_ctx: ModelContext,
    /// Draft model context
    pub draft_ctx: ModelContext,
    /// Configuration
    config: SpeculativeConfig,
    /// Concurrency control
    semaphore: Arc<Semaphore>,
}

impl SpeculativeEngine {
    /// Create a new speculative decoding engine
    ///
    /// # Arguments
    /// * `target_model_path` - Path to main model (e.g., Llama-3.1-8B)
    /// * `draft_model_path` - Path to draft model (e.g., Llama-3.2-1B)
    /// * `config` - Speculative decoding configuration
    ///
    /// # Example
    /// ```ignore
    /// use models::speculative_engine::{SpeculativeEngine, SpeculativeConfig};
    ///
    /// let engine = SpeculativeEngine::new(
    ///     "llama-3.1-8b.gguf",
    ///     "llama-3.2-1b.gguf",
    ///     SpeculativeConfig::default(),
    /// ).await?;
    /// ```
    pub async fn new(
        target_model_path: impl Into<PathBuf>,
        draft_model_path: impl Into<PathBuf>,
        config: SpeculativeConfig,
    ) -> Result<Self> {
        use tokio::task::spawn_blocking;

        // Load models on blocking thread pool (FFI calls)
        let (target_model, draft_model) = spawn_blocking(move || {
            let target_path = target_model_path.into();
            let draft_path = draft_model_path.into();

            // Load target model with GPU offloading
            let target_model = Model::load(&target_path)?;

            // Load draft model (can use fewer GPU layers)
            let draft_model = Model::load(&draft_path)?;

            Ok::<(Model, Model), Error>((target_model, draft_model))
        })
        .await
        .map_err(|e| Error::Backend(format!("Task join error: {}", e)))??;

        let target_model = Arc::new(target_model);
        let draft_model = Arc::new(draft_model);

        // Create contexts with optimized parameters
        let target_params = ContextParams::gpu_optimized();
        let draft_params = ContextParams {
            n_ctx: config.ctx_size_draft as u32,
            n_threads: 4,  // Draft model needs fewer threads
            ..ContextParams::gpu_optimized()
        };

        let target_ctx = ModelContext::new(target_model.clone(), target_params)?;
        let draft_ctx = ModelContext::new(draft_model.clone(), draft_params)?;

        Ok(Self {
            target_model,
            draft_model,
            target_ctx,
            draft_ctx,
            config,
            semaphore: Arc::new(Semaphore::new(1)),
        })
    }

    /// Create with recommended model pairings
    ///
    /// Automatically selects appropriate draft model based on target model
    pub async fn with_pairing(
        target_model_path: impl Into<PathBuf>,
    ) -> Result<Self> {
        let target_path = target_model_path.into();
        let draft_path = Self::recommend_draft_model(&target_path)?;

        Self::new(target_path, draft_path, SpeculativeConfig::default()).await
    }

    /// Recommend draft model based on target model
    fn recommend_draft_model(target_path: &PathBuf) -> Result<PathBuf> {
        let target_name = target_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Model pairings based on common naming patterns
        let draft_name = if target_name.contains("llama-3.1") || target_name.contains("llama31") {
            // Llama 3.1 pairs
            if target_name.contains("8b") || target_name.contains("8B") {
                Some("llama-3.2-1b.gguf")
            } else if target_name.contains("70b") || target_name.contains("70B") {
                Some("llama-3.1-8b.gguf")
            } else {
                Some("llama-3.2-1b.gguf")
            }
        } else if target_name.contains("llama-3") || target_name.contains("llama3") {
            // Llama 3 pairs
            if target_name.contains("70b") {
                Some("llama-3-8b.gguf")
            } else {
                Some("llama-3-2b.gguf")
            }
        } else if target_name.contains("mistral") {
            // Mistral pairs
            if target_name.contains("7b") {
                Some("mistral-0.3.gguf")
            } else if target_name.contains("mixtral") {
                Some("mistral-7b.gguf")
            } else {
                Some("mistral-0.3.gguf")
            }
        } else if target_name.contains("qwen") {
            // Qwen pairs
            if target_name.contains("14b") || target_name.contains("7b") {
                Some("qwen-1.5b.gguf")
            } else if target_name.contains("72b") {
                Some("qwen-7b.gguf")
            } else {
                Some("qwen-1.5b.gguf")
            }
        } else if target_name.contains("deepseek") {
            // DeepSeek pairs
            if target_name.contains("7b") {
                Some("deepseek-1.5b.gguf")
            } else {
                Some("deepseek-1.5b.gguf")
            }
        } else {
            // Default: assume 1B draft model
            None
        };

        // If we can't determine from name, use generic pattern
        let draft_name = draft_name.unwrap_or("draft-1b.gguf");

        // Use same directory as target model
        let draft_path = target_path
            .parent()
            .map(|p| p.join(draft_name))
            .unwrap_or_else(|| PathBuf::from(draft_name));

        Ok(draft_path)
    }

    /// Get model pairing information
    pub fn model_pairing_info(&self) -> ModelPairing {
        ModelPairing {
            target_params: self.target_model.params(),
            draft_params: self.draft_model.params(),
            expected_speedup: self.estimate_speedup(),
        }
    }

    /// Estimate speedup based on model sizes
    fn estimate_speedup(&self) -> f32 {
        let target_params = self.target_model.params().num_params as f32;
        let draft_params = self.draft_model.params().num_params as f32;

        // Speedup formula based on model size ratio
        // Larger draft models give less speedup but better quality
        let ratio = target_params / draft_params;

        if ratio > 50.0 {
            3.0  // 1B draft, 70B target -> 3x speedup
        } else if ratio > 20.0 {
            2.5  // 1B draft, 8B target -> 2.5x speedup
        } else if ratio > 10.0 {
            2.0  // 1B draft, 7B target -> 2x speedup
        } else if ratio > 5.0 {
            1.8  // 1.5B draft, 8B target -> 1.8x speedup
        } else {
            1.5  // Similar sizes -> 1.5x speedup
        }
    }
}

/// Model pairing information
#[derive(Debug, Clone)]
pub struct ModelPairing {
    pub target_params: ModelParams,
    pub draft_params: ModelParams,
    pub expected_speedup: f32,
}

#[derive(Debug, Clone)]
pub struct ModelParams {
    pub num_params: usize,
    pub vocab_size: usize,
    pub context_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.draft_max, 16);
        assert_eq!(config.draft_p_min, 0.8);
    }

    #[test]
    fn test_speculative_config_code() {
        let config = SpeculativeConfig::code_generation();
        assert_eq!(config.draft_max, 16);
        assert_eq!(config.draft_p_min, 0.9);  // Higher threshold
    }

    #[test]
    fn test_speculative_config_creative() {
        let config = SpeculativeConfig::creative();
        assert_eq!(config.draft_max, 8);  // More conservative
        assert_eq!(config.draft_p_min, 0.7);
    }

    #[test]
    fn test_speculative_config_max_speed() {
        let config = SpeculativeConfig::max_speed();
        assert_eq!(config.draft_max, 16);
        assert_eq!(config.draft_p_min, 0.5);  // More aggressive
        assert_eq!(config.ctx_size_draft, 2048);
    }
}
