//! Top-p (nucleus) sampling implementation

use crate::sampler::{Sampler, SamplerType, TokenData};
use core::error::{Error, Result};

/// Top-p (nucleus) sampler
///
/// Selects from smallest set of tokens whose cumulative probability >= p
#[derive(Debug, Clone)]
pub struct TopP {
    p: f32,
    min_keep: usize,
}

impl TopP {
    pub fn new(p: f32) -> Self {
        Self {
            p,
            min_keep: 1,
        }
    }

    pub fn with_min_keep(p: f32, min_keep: usize) -> Self {
        Self {
            p,
            min_keep,
        }
    }
}

impl Sampler for TopP {
    fn sample(&mut self, logits: &mut [TokenData]) -> Result<i32> {
        if self.p >= 1.0 {
            // p >= 1 means no filtering
            let best = logits
                .iter()
                .max_by(|a, b| a.logit.partial_cmp(&b.logit).unwrap())
                .ok_or_else(|| Error::tensor("Empty logits"))?;
            return Ok(best.id);
        }

        if self.p <= 0.0 {
            return Err(Error::tensor("Invalid p value: must be > 0"));
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

        for logit in logits.iter_mut() {
            logit.p /= sum_exp;
        }

        // Sort by probability (descending)
        logits.sort_by(|a, b| b.p.partial_cmp(&a.p).unwrap());

        // Find smallest set with cumulative probability >= p
        let mut cumulative = 0.0f32;
        let mut last_idx = self.min_keep;

        for (i, token) in logits.iter().enumerate() {
            cumulative += token.p;
            if cumulative >= self.p {
                last_idx = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        for i in last_idx..logits.len() {
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
        SamplerType::TopP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_p() {
        let mut sampler = TopP::new(0.9);
        assert_eq!(sampler.p, 0.9);
    }

    #[test]
    fn test_top_p_full() {
        let mut sampler = TopP::new(1.0);
        let mut logits = vec![
            TokenData::new(0, 1.0),
            TokenData::new(1, 2.0),
        ];

        let token = sampler.sample(&mut logits).unwrap();
        assert_eq!(token, 1); // Should pick best
    }
}
