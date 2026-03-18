//! Top-k sampling implementation

use crate::sampler::{Sampler, SamplerType, TokenData};
use barq_core::error::{Error, Result};

/// Top-k sampler
///
/// Only considers the k most likely tokens
#[derive(Debug, Clone)]
pub struct TopK {
    k: i32,
    min_keep: usize,
}

impl TopK {
    pub fn new(k: i32) -> Self {
        Self { k, min_keep: 1 }
    }

    pub fn with_min_keep(k: i32, min_keep: usize) -> Self {
        Self { k, min_keep }
    }
}

impl Sampler for TopK {
    fn sample(&mut self, logits: &mut [TokenData]) -> Result<i32> {
        if self.k <= 0 {
            // k <= 0 means no filtering
            // Return greedy (most likely token)
            let best = logits
                .iter()
                .max_by(|a, b| a.logit.partial_cmp(&b.logit).unwrap())
                .ok_or_else(|| Error::tensor("Empty logits"))?;
            return Ok(best.id);
        }

        let k = self.k as usize;

        if k >= logits.len() {
            // k >= vocab size means no filtering
            let best = logits
                .iter()
                .max_by(|a, b| a.logit.partial_cmp(&b.logit).unwrap())
                .ok_or_else(|| Error::tensor("Empty logits"))?;
            return Ok(best.id);
        }

        // Sort by logit (descending)
        logits.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap());

        // Zero out tokens beyond top k
        for i in k..logits.len() {
            logits[i].logit = f32::NEG_INFINITY;
        }

        // Return best token
        Ok(logits[0].id)
    }

    fn reset(&mut self) {
        // No state to reset
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::TopK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k() {
        let mut sampler = TopK::new(2);
        let mut logits = vec![
            TokenData::new(0, 1.0),
            TokenData::new(1, 3.0),
            TokenData::new(2, 2.0),
            TokenData::new(3, 0.5),
        ];

        let token = sampler.sample(&mut logits).unwrap();
        assert_eq!(token, 1); // Top token
    }

    #[test]
    fn test_top_k_zero() {
        let mut sampler = TopK::new(0);
        let mut logits = vec![TokenData::new(0, 1.0), TokenData::new(1, 3.0)];

        let token = sampler.sample(&mut logits).unwrap();
        assert_eq!(token, 1); // Greedy
    }
}
