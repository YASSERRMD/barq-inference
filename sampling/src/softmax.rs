//! Softmax-based sampling completion

use crate::sampler::{Sampler, SamplerType, TokenData};
use crate::temperature::Temperature;
use barq_core::error::{Error, Result};

/// Softmax sampling
pub struct SoftmaxSampler;

impl SoftmaxSampler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SoftmaxSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for SoftmaxSampler {
    fn sample(&mut self, logits: &mut [TokenData]) -> Result<i32> {
        if logits.is_empty() {
            return Err(Error::tensor("Empty logits for sampling"));
        }

        // Find max logit for numerical stability
        let max_logit = logits.iter()
            .map(|t| t.logit)
            .fold(f32::NEG_INFINITY, |acc, x| acc.max(x));

        // Compute softmax probabilities
        let mut sum_exp = 0.0f32;
        for logit in logits.iter_mut() {
            logit.p = (logit.logit - max_logit).exp();
            sum_exp += logit.p;
        }

        // Normalize
        for logit in logits.iter_mut() {
            logit.p /= sum_exp;
        }

        // Sample from distribution
        let r: f32 = rand::random();
        let mut cumulative = 0.0f32;

        for (i, token) in logits.iter().enumerate() {
            cumulative += token.p;
            if r <= cumulative {
                return Ok(token.id);
            }
        }

        // Fallback to last token
        Ok(logits.last().unwrap().id)
    }

    fn reset(&mut self) {
        // No state to reset
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(Self::new())
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Custom
    }
}
