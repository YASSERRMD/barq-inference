//! Speculative decoding verification loop
//!
//! Implements the core speculative decoding algorithm:
//! 1. Draft model predicts k tokens ahead
//! 2. Main model verifies all k tokens in parallel
//! 3. Accept tokens where main model agrees with draft
//! 4. Resample from main model distribution at first rejection
//! 5. Repeat until max tokens or EOS

use std::time::{Duration, Instant};
use barq_core::error::{Error, Result};
use barq_core::tensor::Tensor;

use crate::speculative_engine::SpeculativeEngine;
use crate::speculative_engine::SpeculativeConfig;

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
    /// Speculation rounds (iterations)
    pub rounds: usize,
    /// Average acceptance rate
    pub acceptance_rate: f32,
    /// Time spent in draft model (ms)
    pub draft_time_ms: f64,
    /// Time spent in verification (ms)
    pub verify_time_ms: f64,
    /// Total time (ms)
    pub total_time_ms: f64,
}

impl SpeculativeStats {
    pub fn update_acceptance_rate(&mut self) {
        let total = self.accepted_tokens + self.resampled_tokens;
        self.acceptance_rate = if total > 0 {
            self.accepted_tokens as f32 / total as f64
        } else {
            0.0
        };
    }

    pub fn print(&self) {
        println!("\n=== Speculative Decoding Stats ===");
        println!("Total tokens:      {}", self.total_tokens);
        println!("Accepted tokens:   {}", self.accepted_tokens);
        println!("Resampled tokens:  {}", self.resampled_tokens);
        println!("Rounds:            {}", self.rounds);
        println!("Acceptance rate:   {:.2}%", self.acceptance_rate * 100.0);
        println!("Draft time:        {:.2} ms", self.draft_time_ms);
        println!("Verify time:       {:.2} ms", self.verify_time_ms);
        println!("Total time:        {:.2} ms", self.total_time_ms);
        println!("Effective TPS:     {:.2} tok/s", self.effective_tps());
        println!("==================================\n");
    }

    pub fn effective_tps(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.total_tokens as f64) / (self.total_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

impl SpeculativeEngine {
    /// Generate tokens with speculative decoding
    ///
    /// # Arguments
    /// * `prompt_tokens` - Input prompt token IDs
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Returns
    /// Generated tokens and statistics
    pub fn generate_speculative(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
    ) -> Result<(Vec<i32>, SpeculativeStats)> {
        let _permit = std::sync::Arc::clone(&self.semaphore)
            .try_acquire()
            .map_err(|_| Error::Backend("Speculative decoding already in progress".to_string()))?;

        let start_time = Instant::now();
        let mut output = Vec::new();
        let mut current_tokens = prompt_tokens.to_vec();
        let mut stats = SpeculativeStats::default();

        while output.len() < max_tokens {
            stats.rounds += 1;

            // Step 1: Draft model predicts k tokens ahead
            let draft_start = Instant::now();
            let draft_tokens = self.draft_speculate(&current_tokens, max_tokens - output.len())?;
            stats.draft_time_ms += draft_start.elapsed().as_secs_f64() * 1000.0;

            if draft_tokens.is_empty() {
                break;
            }

            // Step 2: Main model verifies and accepts tokens
            let verify_start = Instant::now();
            let verified_tokens = self.verify_and_accept(&current_tokens, &draft_tokens, &mut stats)?;
            stats.verify_time_ms += verify_start.elapsed().as_secs_f64() * 1000.0;

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

        stats.total_tokens = output.len();
        stats.total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        stats.update_acceptance_rate();

        Ok((output, stats))
    }

    /// Draft model speculation: predict k tokens ahead
    fn draft_speculate(
        &self,
        context: &[i32],
        remaining: usize,
    ) -> Result<Vec<TokenWithProb>> {
        let k = self.config.draft_max.min(remaining);
        let mut draft_tokens = Vec::new();
        let mut current_context = context.to_vec();

        for _ in 0..k {
            // Get draft model prediction
            let logits = self.draft_ctx.forward(&current_context)?;

            // Sample token from draft distribution
            let sampled_token = sample_token(&logits, 0.8)?; // Use draft temperature

            // Get probability for this token
            let prob = logits.get(sampled_token as usize)
                .copied()
                .unwrap_or(0.0);

            draft_tokens.push(TokenWithProb {
                token: sampled_token,
                prob,
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
    fn verify_and_accept(
        &self,
        context: &[i32],
        draft_tokens: &[TokenWithProb],
        stats: &mut SpeculativeStats,
    ) -> Result<Vec<i32>> {
        let mut accepted = Vec::new();
        let mut verify_context = context.to_vec();

        for draft_token in draft_tokens {
            // Main model prediction for this position
            let main_logits = self.target_ctx.forward(&verify_context)?;

            if main_logits.is_empty() {
                break;
            }

            // Sample from main model distribution
            let main_token = sample_token(&main_logits, 1.0)?; // Temperature = 1.0

            // Compute acceptance probability
            // q(x) / p(x) where q is main model, p is draft model
            let main_prob = main_logits.get(main_token as usize)
                .copied()
                .unwrap_or(0.0);

            let accept_prob = if main_token == draft_token.token {
                // Both models agree - compute ratio of probabilities
                let ratio = main_prob / (draft_token.prob + 1e-10);
                ratio.min(1.0)
            } else {
                // Models disagree
                0.0
            };

            // Accept or reject based on threshold
            if accept_prob >= self.config.draft_p_min {
                // Accept draft token
                accepted.push(draft_token.token);
                verify_context.push(draft_token.token);
                stats.accepted_tokens += 1;
            } else {
                // Reject and resample from main model
                accepted.push(main_token);
                verify_context.push(main_token);
                stats.resampled_tokens += 1;
                break; // Stop speculation after rejection
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

    // Fallback to most likely token
    probs.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i32)
        .ok_or_else(|| Error::tensor("Failed to sample token"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_stats_default() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
    }

    #[test]
    fn test_speculative_stats_update() {
        let mut stats = SpeculativeStats::default();
        stats.accepted_tokens = 80;
        stats.resampled_tokens = 20;
        stats.update_acceptance_rate();
        assert!((stats.acceptance_rate - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_speculative_stats_tps() {
        let mut stats = SpeculativeStats::default();
        stats.total_tokens = 100;
        stats.total_time_ms = 1000.0; // 1 second
        assert_eq!(stats.effective_tps(), 100.0);
    }

    #[test]
    fn test_sample_token() {
        let logits = vec![0.1, 0.2, 0.3, 0.4];
        let token = sample_token(&logits, 1.0).unwrap();
        assert!(token >= 0 && token < 4);
    }

    #[test]
    fn test_sample_token_temperature() {
        let logits = vec![0.1, 0.2, 0.3, 0.4];
        let token_cold = sample_token(&logits, 0.1).unwrap();
        let token_hot = sample_token(&logits, 2.0).unwrap();

        // Both should be valid tokens
        assert!(token_cold >= 0 && token_cold < 4);
        assert!(token_hot >= 0 && token_hot < 4);
    }
}
