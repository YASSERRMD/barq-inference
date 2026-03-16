//! Penalty implementations for sampling

use barq_core::error::{Error, Result};

/// Repetition penalty sampler
#[derive(Debug, Clone)]
pub struct RepetitionPenalty {
    penalty: f32,
}

impl RepetitionPenalty {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl crate::sampler::Sampler for RepetitionPenalty {
    fn sample(&mut self, logits: &mut [crate::sampler::TokenData]) -> Result<i32> {
        // This modifies logits in place but doesn't sample
        // Typically used in a chain
        for logit in logits.iter_mut() {
            logit.logit /= self.penalty;
        }
        Err(Error::tensor("RepetitionPenalty must be used in a chain"))
    }

    fn reset(&mut self) {}

    fn clone_box(&self) -> Box<dyn crate::sampler::Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> crate::sampler::SamplerType {
        crate::sampler::SamplerType::RepetitionPenalty
    }
}

/// Frequency penalty sampler
#[derive(Debug, Clone)]
pub struct FrequencyPenalty {
    penalty: f32,
}

impl FrequencyPenalty {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl crate::sampler::Sampler for FrequencyPenalty {
    fn sample(&mut self, logits: &mut [crate::sampler::TokenData]) -> Result<i32> {
        for logit in logits.iter_mut() {
            logit.logit -= self.penalty;
        }
        Err(Error::tensor("FrequencyPenalty must be used in a chain"))
    }

    fn reset(&mut self) {}

    fn clone_box(&self) -> Box<dyn crate::sampler::Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> crate::sampler::SamplerType {
        crate::sampler::SamplerType::FrequencyPenalty
    }
}

/// Presence penalty sampler
#[derive(Debug, Clone)]
pub struct PresencePenalty {
    penalty: f32,
}

impl PresencePenalty {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl crate::sampler::Sampler for PresencePenalty {
    fn sample(&mut self, logits: &mut [crate::sampler::TokenData]) -> Result<i32> {
        for logit in logits.iter_mut() {
            logit.logit -= self.penalty;
        }
        Err(Error::tensor("PresencePenalty must be used in a chain"))
    }

    fn reset(&mut self) {}

    fn clone_box(&self) -> Box<dyn crate::sampler::Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> crate::sampler::SamplerType {
        crate::sampler::SamplerType::PresencePenalty
    }
}

/// Min-P sampling sampler
#[derive(Debug, Clone)]
pub struct MinP {
    min_p: f32,
}

impl MinP {
    pub fn new(min_p: f32) -> Self {
        Self { min_p }
    }
}

impl crate::sampler::Sampler for MinP {
    fn sample(&mut self, logits: &mut [crate::sampler::TokenData]) -> Result<i32> {
        let min_p = self.min_p;
        if min_p <= 0.0 || min_p >= 1.0 {
            return Ok(logits[0].id);
        }

        // Compute softmax
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

        // Filter by min_p
        for logit in logits.iter_mut() {
            if logit.p < min_p {
                logit.logit = f32::NEG_INFINITY;
            }
        }

        Err(Error::tensor("MinP must be used in a chain"))
    }

    fn reset(&mut self) {}

    fn clone_box(&self) -> Box<dyn crate::sampler::Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> crate::sampler::SamplerType {
        crate::sampler::SamplerType::MinP
    }
}

/// Mirostat sampling type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirostatType {
    Mirostat,
    Mirostat2,
}

/// Mirostat sampling sampler
#[derive(Debug, Clone)]
pub struct Mirostat {
    tau: f32,
    eta: f32,
    mu: f32,
    mirostat_type: MirostatType,
}

impl Mirostat {
    pub fn new(tau: f32, eta: f32, mirostat_type: MirostatType) -> Self {
        Self {
            tau,
            eta,
            mu: 0.0,
            mirostat_type,
        }
    }
}

impl crate::sampler::Sampler for Mirostat {
    fn sample(&mut self, logits: &mut [crate::sampler::TokenData]) -> Result<i32> {
        // Compute softmax
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

        // Compute surprise
        let mut surprise = 0.0f32;
        for logit in logits.iter() {
            if logit.p > 0.0 {
                surprise -= logit.p.log2();
            }
        }

        // Update mu
        self.mu = self.mu * (1.0 - self.eta) + surprise * self.eta;

        // Adjust logits
        let scaling_factor = self.mu / self.tau;
        for logit in logits.iter_mut() {
            logit.logit *= scaling_factor;
        }

        Err(Error::tensor("Mirostat must be used in a chain"))
    }

    fn reset(&mut self) {
        self.mu = 0.0;
    }

    fn clone_box(&self) -> Box<dyn crate::sampler::Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> crate::sampler::SamplerType {
        crate::sampler::SamplerType::Mirostat
    }
}

/// Typical sampling sampler
#[derive(Debug, Clone)]
pub struct Typical {
    typical_p: f32,
}

impl Typical {
    pub fn new(typical_p: f32) -> Self {
        Self { typical_p }
    }
}

impl crate::sampler::Sampler for Typical {
    fn sample(&mut self, logits: &mut [crate::sampler::TokenData]) -> Result<i32> {
        let typical_p = self.typical_p;
        if typical_p <= 0.0 || typical_p >= 1.0 {
            return Ok(logits[0].id);
        }

        // Compute softmax
        let mut max_logit = f32::NEG_INFINITY;
        for logit in logits.iter() {
            max_logit = max_logit.max(logit.logit);
        }

        let mut probs: Vec<f32> = logits.iter()
            .map(|x| (x.logit - max_logit).exp())
            .collect();

        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Filter by typical probability
        for (i, &p) in probs.iter().enumerate() {
            if p < typical_p {
                logits[i].logit = f32::NEG_INFINITY;
            }
        }

        Err(Error::tensor("Typical must be used in a chain"))
    }

    fn reset(&mut self) {}

    fn clone_box(&self) -> Box<dyn crate::sampler::Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> crate::sampler::SamplerType {
        crate::sampler::SamplerType::Typical
    }
}

/// XTC (eXtending The Context) sampling sampler
#[derive(Debug, Clone)]
pub struct XtcSampler {
    threshold: f32,
}

impl XtcSampler {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl crate::sampler::Sampler for XtcSampler {
    fn sample(&mut self, logits: &mut [crate::sampler::TokenData]) -> Result<i32> {
        // XTC filtering - removes tokens above threshold
        for logit in logits.iter_mut() {
            if logit.p > self.threshold {
                logit.logit = f32::NEG_INFINITY;
            }
        }

        Err(Error::tensor("XtcSampler must be used in a chain"))
    }

    fn reset(&mut self) {}

    fn clone_box(&self) -> Box<dyn crate::sampler::Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> crate::sampler::SamplerType {
        crate::sampler::SamplerType::Xtc
    }
}

/// Apply repetition penalty to logits
pub fn apply_repetition_penalty(logits: &mut [f32], tokens: &[i32], penalty: f32) {
    if penalty <= 0.0 || tokens.is_empty() {
        return;
    }

    for &token in tokens {
        if token >= 0 && (token as usize) < logits.len() {
            logits[token as usize] /= penalty;
        }
    }
}

/// Apply frequency penalty to logits
pub fn apply_frequency_penalty(logits: &mut [f32], token_counts: &[usize], penalty: f32) {
    if penalty <= 0.0 {
        return;
    }

    for (i, &count) in token_counts.iter().enumerate() {
        if i < logits.len() && count > 0 {
            logits[i] -= penalty * count as f32;
        }
    }
}

/// Apply presence penalty to logits
pub fn apply_presence_penalty(logits: &mut [f32], tokens: &[i32], penalty: f32) {
    if penalty <= 0.0 || tokens.is_empty() {
        return;
    }

    for &token in tokens {
        if token >= 0 && (token as usize) < logits.len() {
            logits[token as usize] -= penalty;
        }
    }
}

/// Min-P sampling: filter tokens below minimum probability
pub fn apply_min_p(logits: &mut [f32], min_p: f32) -> Result<()> {
    if min_p <= 0.0 || min_p >= 1.0 {
        return Ok(());
    }

    // Compute softmax
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let mut probs: Vec<f32> = logits.iter()
        .map(|x| (x - max_logit).exp())
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    // Filter by min_p
    let min_prob = min_p;
    for (i, &p) in probs.iter().enumerate() {
        if p < min_prob {
            logits[i] = f32::NEG_INFINITY;
        }
    }

    Ok(())
}

/// Mirostat sampling parameters
#[derive(Debug, Clone)]
pub struct MirostatSampler {
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
}

impl MirostatSampler {
    pub fn new(tau: f32, eta: f32) -> Self {
        Self {
            tau,
            eta,
            mu: 0.0,
        }
    }
}

impl MirostatSampler {
    pub fn apply(&mut self, logits: &mut [f32]) -> Result<()> {
        // Compute softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let mut probs: Vec<f32> = logits.iter()
            .map(|x| (x - max_logit).exp())
            .collect();

        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Surprise = -log2(prob)
        let mut surprise = 0.0f32;
        for &p in probs.iter() {
            if p > 0.0 {
                surprise -= p.log2();
            }
        }

        // Update mu
        self.mu = self.mu * (1.0 - self.eta) + surprise * self.eta;

        // Adjust logits to maintain target surprise (tau)
        let scaling_factor = self.mu / self.tau;

        for logit in logits.iter_mut() {
            *logit *= scaling_factor;
        }

        Ok(())
    }
}

/// Typical sampling: maintain similar probability mass
pub fn apply_typical_sampling(logits: &mut [f32], typical_p: f32) -> Result<()> {
    if typical_p <= 0.0 || typical_p >= 1.0 {
        return Ok(());
    }

    // Compute softmax
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let probs: Vec<f32> = logits.iter()
        .map(|x| (x - max_logit).exp())
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let probs_sum = sum;
        let mut probs_normalized = probs.clone();
        for p in probs_normalized.iter_mut() {
            *p /= probs_sum;
        }
    }

    // Filter by typical probability
    let typical_threshold = typical_p;
    for (i, &p) in probs.iter().enumerate() {
        if p < typical_threshold {
            logits[i] = f32::NEG_INFINITY;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert!((logits[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_min_p() {
        let mut logits = vec![0.1, 0.2, 0.7];
        assert!(apply_min_p(&mut logits, 0.3).is_ok());
    }
}

