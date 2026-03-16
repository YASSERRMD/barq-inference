//! Temperature sampling implementation

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::sampler::{Sampler, SamplerType, TokenData};
use core::error::{Error, Result};

/// Temperature sampler
///
/// Higher temperature = more random, lower = more deterministic
#[derive(Debug, Clone)]
pub struct Temperature {
    temperature: f32,
    rng: ChaCha8Rng,
}

impl Temperature {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            rng: ChaCha8Rng::from_entropy(),
        }
    }

    pub fn with_seed(temperature: f32, seed: u32) -> Self {
        Self {
            temperature,
            rng: rand_chacha::ChaCha8Rng::seed_from_u64(seed as u64),
        }
    }
}

impl Sampler for Temperature {
    fn sample(&mut self, logits: &mut [TokenData]) -> Result<i32> {
        if self.temperature <= 0.0 {
            // Zero temperature = greedy
            let best = logits
                .iter()
                .max_by(|a, b| a.logit.partial_cmp(&b.logit).unwrap())
                .ok_or_else(|| Error::tensor("Empty logits"))?;
            return Ok(best.id);
        }

        // Apply temperature scaling
        for logit in logits.iter_mut() {
            logit.logit /= self.temperature;
        }

        // Sample from scaled distribution
        let mut max_logit = f32::NEG_INFINITY;
        for logit in logits.iter() {
            max_logit = max_logit.max(logit.logit);
        }

        let mut sum_exp = 0.0f32;
        for logit in logits.iter_mut() {
            logit.p = (logit.logit - max_logit).exp();
            sum_exp += logit.p;
        }

        for logit in logits.iter_mut() {
            logit.p /= sum_exp;
        }

        let r: f32 = self.rng.gen();
        let mut cumulative = 0.0f32;
        for token in logits.iter() {
            cumulative += token.p;
            if r <= cumulative {
                return Ok(token.id);
            }
        }

        Ok(logits.last().ok_or_else(|| Error::tensor("No tokens"))?.id)
    }

    fn reset(&mut self) {
        self.rng = ChaCha8Rng::from_entropy();
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(Self {
            temperature: self.temperature,
            rng: ChaCha8Rng::from_entropy(),
        })
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Temperature
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature() {
        let sampler = Temperature::new(0.8);
        assert_eq!(sampler.temperature, 0.8);
    }

    #[test]
    fn test_temperature_zero() {
        let mut sampler = Temperature::new(0.0);
        let mut logits = vec![
            TokenData::new(0, -1.0),
            TokenData::new(1, 2.0),
            TokenData::new(2, 1.0),
        ];

        let token = sampler.sample(&mut logits).unwrap();
        assert_eq!(token, 1); // Should be greedy
    }
}
