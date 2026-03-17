//! Core sampler trait and types

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use barq_core::error::{Error, Result};

/// Token data for sampling
#[derive(Debug, Clone, Copy)]
pub struct TokenData {
    /// Token ID
    pub id: i32,
    /// Logit value (log-odds)
    pub logit: f32,
    /// Probability
    pub p: f32,
}

impl TokenData {
    pub fn new(id: i32, logit: f32) -> Self {
        Self { id, logit, p: 0.0 }
    }
}

/// Sampler trait
pub trait Sampler: Send + Sync {
    /// Apply sampling to logits
    fn sample(&mut self, logits: &mut [TokenData]) -> Result<i32>;

    /// Reset sampler state
    fn reset(&mut self);

    /// Clone the sampler
    fn clone_box(&self) -> Box<dyn Sampler>;

    /// Returns the sampler type
    fn sampler_type(&self) -> SamplerType;
}

/// Sampler type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerType {
    /// Temperature sampling
    Temperature,
    /// Top-k sampling
    TopK,
    /// Top-p (nucleus) sampling
    TopP,
    /// Min-p sampling
    MinP,
    /// Mirostat sampling
    Mirostat,
    /// Typical sampling
    Typical,
    /// Greedy sampling
    Greedy,
    /// Repetition penalty
    RepetitionPenalty,
    /// Frequency penalty
    FrequencyPenalty,
    /// Presence penalty
    PresencePenalty,
    /// XTC sampling
    Xtc,
    /// Custom sampler
    Custom,
}

impl Clone for Box<dyn Sampler> {
    fn clone(&self) -> Box<dyn Sampler> {
        self.clone_box()
    }
}

/// Greedy sampler (selects token with highest probability)
#[derive(Debug, Clone)]
pub struct GreedySampler;

impl GreedySampler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GreedySampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &mut [TokenData]) -> Result<i32> {
        if logits.is_empty() {
            return Err(Error::tensor("Empty logits for sampling"));
        }

        // Find token with highest logit
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.logit.partial_cmp(&b.1.logit).unwrap())
            .ok_or_else(|| Error::tensor("Failed to find max token"))?;

        Ok(best.1.id)
    }

    fn reset(&mut self) {
        // No state to reset
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Greedy
    }
}

/// Dist (distribution) sampler - samples from probability distribution
#[derive(Debug, Clone)]
pub struct DistSampler {
    rng: ChaCha8Rng,
}

impl DistSampler {
    pub fn new(seed: u32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed as u64),
        }
    }

    pub fn from_rng(rng: ChaCha8Rng) -> Self {
        Self { rng }
    }
}

impl Sampler for DistSampler {
    fn sample(&mut self, logits: &mut [TokenData]) -> Result<i32> {
        if logits.is_empty() {
            return Err(Error::tensor("Empty logits for sampling"));
        }

        // Compute softmax probabilities
        let mut max_logit = f32::NEG_INFINITY;
        for logit in logits.iter() {
            max_logit = max_logit.max(logit.logit);
        }

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
        let r: f32 = self.rng.gen();
        let mut cumulative = 0.0f32;
        for token in logits.iter() {
            cumulative += token.p;
            if r <= cumulative {
                return Ok(token.id);
            }
        }

        // Fallback to last token
        Ok(logits.last().ok_or_else(|| Error::tensor("No tokens"))?.id)
    }

    fn reset(&mut self) {
        // Reset RNG with new seed
        self.rng = ChaCha8Rng::from_entropy();
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        // Can't clone RNG, create new one
        Box::new(DistSampler::from_rng(ChaCha8Rng::from_entropy()))
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Custom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_data() {
        let token = TokenData::new(42, 1.5);
        assert_eq!(token.id, 42);
        assert_eq!(token.logit, 1.5);
        assert_eq!(token.p, 0.0);
    }

    #[test]
    fn test_greedy_sampler() {
        let mut sampler = GreedySampler::new();
        let mut logits = vec![
            TokenData::new(0, -1.0),
            TokenData::new(1, 2.0),
            TokenData::new(2, 1.0),
        ];

        let token = sampler.sample(&mut logits).unwrap();
        assert_eq!(token, 1); // Highest logit
    }
}
