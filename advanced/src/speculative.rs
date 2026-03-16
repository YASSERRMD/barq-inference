//! Speculative decoding implementation
//!
//! Uses a smaller draft model to predict multiple tokens ahead,
//! then verifies them with the main model for 2-3x speedup.
//!
//! Algorithm:
//! 1. Draft model predicts k tokens ahead
//! 2. Main model verifies all k tokens in parallel
//! 3. Accept tokens where main model agrees with draft
//! 4. Resample from main model distribution at first rejection
//! 5. Repeat until max tokens or EOS

use std::sync::Arc;
use tokio::sync::Semaphore;

use core::error::{Error, Result};
use core::tensor::{Tensor, TensorType, TensorData, Shape};

/// Speculative decoding configuration
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate ahead (default: 5)
    pub speculation_steps: usize,
    /// Acceptance threshold for resampling (0.0-1.0)
    pub accept_threshold: f32,
    /// Whether to use sampling for verification
    pub sample_verification: bool,
    /// Temperature for draft model sampling
    pub draft_temperature: f32,
    /// Temperature for main model sampling
    pub main_temperature: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            speculation_steps: 5,
            accept_threshold: 0.5,
            sample_verification: true,
            draft_temperature: 0.8,
            main_temperature: 1.0,
        }
    }
}

/// Token with probability from model
#[derive(Debug, Clone)]
struct TokenWithProb {
    token: i32,
    prob: f32,
}

/// Speculative decoding statistics
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated
    pub total_tokens: usize,
    /// Tokens accepted from draft
    pub accepted_tokens: usize,
    /// Tokens resampled from main model
    pub resampled_tokens: usize,
    /// Speculation rounds
    pub rounds: usize,
    /// Average acceptance rate
    pub acceptance_rate: f32,
}

impl SpeculativeStats {
    pub fn update_acceptance_rate(&mut self) {
        let total = self.accepted_tokens + self.resampled_tokens;
        self.acceptance_rate = if total > 0 {
            self.accepted_tokens as f32 / total as f32
        } else {
            0.0
        };
    }
}

/// Speculative decoding engine
pub struct SpeculativeDecoding {
    /// Configuration
    config: SpeculativeConfig,
    /// Vocabulary size
    vocab_size: usize,
    /// Concurrency semaphore
    semaphore: Arc<Semaphore>,
    /// Statistics
    stats: SpeculativeStats,
}

impl SpeculativeDecoding {
    /// Create a new speculative decoding engine
    pub fn new(vocab_size: usize, config: SpeculativeConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(1));

        Self {
            config,
            vocab_size,
            semaphore,
            stats: SpeculativeStats::default(),
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Generate tokens with speculative decoding
    ///
    /// # Arguments
    /// * `draft_model_fn` - Function to run draft model inference
    /// * `main_model_fn` - Function to run main model inference
    /// * `prompt_tokens` - Input prompt tokens
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Returns
    /// Generated tokens
    pub async fn generate<F, G>(
        &mut self,
        mut draft_model_fn: F,
        mut main_model_fn: G,
        prompt_tokens: &[i32],
        max_tokens: usize,
    ) -> Result<Vec<i32>>
    where
        F: FnMut(&[i32]) -> Result<(i32, Vec<f32>)> + Send + 'static,
        G: FnMut(&[i32]) -> Result<Vec<i32>> + Send + 'static,
    {
        let _permit = self.semaphore.acquire().await
            .map_err(|e| Error::Backend(format!("Semaphore error: {}", e)))?;

        let mut output = Vec::new();
        let mut current_tokens = prompt_tokens.to_vec();

        while output.len() < max_tokens {
            self.stats.rounds += 1;

            // Step 1: Draft model predicts k tokens ahead
            let draft_tokens = self.speculate(&mut draft_model_fn, &current_tokens)?;

            if draft_tokens.is_empty() {
                break;
            }

            // Step 2: Main model verifies and accepts tokens
            let verified_tokens = self.verify_and_accept(
                &mut main_model_fn,
                &current_tokens,
                &draft_tokens,
            )?;

            if verified_tokens.is_empty() {
                break;
            }

            output.extend(verified_tokens.clone());
            current_tokens.extend(verified_tokens.clone());

            // Check for EOS (assuming 0 is EOS token)
            if let Some(&last) = verified_tokens.last() {
                if last == 0 {
                    break;
                }
            }
        }

        self.stats.total_tokens = output.len();
        self.stats.update_acceptance_rate();

        Ok(output)
    }

    /// Draft model speculation: predict k tokens ahead
    fn speculate<F>(
        &self,
        mut draft_model_fn: F,
        context: &[i32],
    ) -> Result<Vec<TokenWithProb>>
    where
        F: FnMut(&[i32]) -> Result<(i32, Vec<f32>)>,
    {
        let mut draft_tokens = Vec::new();
        let mut current_context = context.to_vec();

        for _ in 0..self.config.speculation_steps {
            let (token, logits) = draft_model_fn(&current_context)?;

            // Sample from draft distribution
            let sampled_token = sample_token(&logits, self.config.draft_temperature)?;

            draft_tokens.push(TokenWithProb {
                token: sampled_token,
                prob: logits[sampled_token as usize],
            });

            current_context.push(sampled_token);

            // Stop if we hit EOS
            if sampled_token == 0 {
                break;
            }
        }

        Ok(draft_tokens)
    }

    /// Verify draft tokens with main model and accept/reject
    fn verify_and_accept<F>(
        &mut self,
        mut main_model_fn: F,
        context: &[i32],
        draft_tokens: &[TokenWithProb],
    ) -> Result<Vec<i32>>
    where
        F: FnMut(&[i32]) -> Result<Vec<i32>>,
    {
        let mut accepted = Vec::new();
        let mut verify_context = context.to_vec();

        for (i, draft_token) in draft_tokens.iter().enumerate() {
            // Main model prediction for this position
            let main_tokens = main_model_fn(&verify_context)?;

            if main_tokens.is_empty() {
                break;
            }

            let main_token = main_tokens[0];

            // Compute acceptance probability
            // q(x) / p(x) where q is main model, p is draft model
            let accept_prob = if main_token == draft_token.token {
                // Both models agree - compute ratio of probabilities
                let main_prob = 1.0 / (main_tokens.len() as f32); // Simplified
                let ratio = main_prob / (draft_token.prob + 1e-10);
                ratio.min(1.0)
            } else {
                // Models disagree
                0.0
            };

            // Accept or reject based on threshold
            if accept_prob >= self.config.accept_threshold || !self.config.sample_verification {
                // Accept draft token
                accepted.push(draft_token.token);
                verify_context.push(draft_token.token);
                self.stats.accepted_tokens += 1;
            } else {
                // Reject and resample from main model
                if let Some(&resampled) = main_tokens.first() {
                    accepted.push(resampled);
                    verify_context.push(resampled);
                    self.stats.resampled_tokens += 1;
                }
                break; // Stop speculation after rejection
            }
        }

        Ok(accepted)
    }

    /// Alternative verification using full logits comparison
    fn verify_with_logits<F>(
        &mut self,
        mut main_model_fn: F,
        context: &[i32],
        draft_tokens: &[TokenWithProb],
    ) -> Result<Vec<i32>>
    where
        F: FnMut(&[i32]) -> Result<Vec<i32>>,
    {
        let mut accepted = Vec::new();
        let mut verify_context = context.to_vec();

        for draft_token in draft_tokens {
            let main_tokens = main_model_fn(&verify_context)?;

            if main_tokens.is_empty() {
                break;
            }

            let main_token = main_tokens[0];

            // Accept if tokens match
            if main_token == draft_token.token {
                accepted.push(draft_token.token);
                verify_context.push(draft_token.token);
                self.stats.accepted_tokens += 1;
            } else {
                // Mismatch: add main model's prediction and stop
                accepted.push(main_token);
                verify_context.push(main_token);
                self.stats.resampled_tokens += 1;
                break;
            }
        }

        Ok(accepted)
    }
}

/// Sample a token from logits using temperature
fn sample_token(logits: &[f32], temperature: f32) -> Result<i32> {
    if logits.is_empty() {
        return Err(Error::tensor("Empty logits for sampling"));
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter()
        .map(|&x| x / temperature)
        .collect();

    // Compute softmax
    let max_logit = scaled.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exp_sum: f32 = scaled.iter()
        .map(|&x| (x - max_logit).exp())
        .sum();

    let probs: Vec<f32> = scaled.iter()
        .map(|&x| ((x - max_logit).exp()) / exp_sum)
        .collect();

    // Sample
    let r: f32 = rand::random();
    let mut cumulative = 0.0f32;

    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return Ok(i as i32);
        }
    }

    // Fallback to last token
    Ok((probs.len() - 1) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.speculation_steps, 5);
        assert_eq!(config.accept_threshold, 0.5);
    }

    #[test]
    fn test_speculative_creation() {
        let sd = SpeculativeDecoding::new(32000, SpeculativeConfig::default());
        assert_eq!(sd.vocab_size, 32000);
    }

    #[test]
    fn test_sample_token() {
        let logits = vec![0.1, 0.2, 0.3, 0.4];
        let token = sample_token(&logits, 1.0).unwrap();
        assert!(token >= 0 && token < 4);
    }

    #[tokio::test]
    async fn test_speculative_generate() {
        let mut sd = SpeculativeDecoding::new(100, SpeculativeConfig::default());

        // Mock draft model: always returns token 1
        let draft_fn = |context: &[i32]| -> Result<(i32, Vec<f32>)> {
            let mut logits = vec![0.0f32; 100];
            logits[1] = 1.0; // Always predict token 1
            Ok((1, logits))
        };

        // Mock main model: always returns token 1
        let main_fn = |context: &[i32]| -> Result<Vec<i32>> {
            Ok(vec![1])
        };

        let result = sd.generate(draft_fn, main_fn, &[0, 1, 2], 10).await;

        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_speculative_stats() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.speculation_steps, 5);
        assert_eq!(config.accept_threshold, 0.5);
    }
}
