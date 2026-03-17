//! Sampler chain for composing multiple samplers

use std::sync::{Arc, Mutex};

use crate::sampler::{Sampler, SamplerType, TokenData};
use barq_core::error::{Error, Result};

/// Sampler chain
///
/// Applies multiple samplers in sequence
pub struct SamplerChain {
    samplers: Vec<Box<dyn Sampler>>,
}

impl SamplerChain {
    pub fn new() -> Self {
        Self {
            samplers: Vec::new(),
        }
    }

    pub fn add(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.samplers.push(sampler);
        self
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samplers: Vec::with_capacity(capacity),
        }
    }
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for SamplerChain {
    fn sample(&mut self, logits: &mut [TokenData]) -> Result<i32> {
        // Apply all samplers except the last (which should do the actual sampling)
        let n_samplers = self.samplers.len();

        if n_samplers == 0 {
            return Err(Error::tensor("Empty sampler chain"));
        }

        for i in 0..n_samplers.saturating_sub(1) {
            self.samplers[i].sample(logits)?;
        }

        // Last sampler does the actual selection
        self.samplers[n_samplers - 1].sample(logits)
    }

    fn reset(&mut self) {
        for sampler in self.samplers.iter_mut() {
            sampler.reset();
        }
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        // Can't easily clone chain of trait objects
        // Return new empty chain
        Box::new(SamplerChain::new())
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Custom
    }
}

/// Shared sampler chain with thread-safe access
pub struct SharedSamplerChain {
    chain: Arc<Mutex<SamplerChain>>,
}

impl Default for SharedSamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedSamplerChain {
    pub fn new() -> Self {
        Self {
            chain: Arc::new(Mutex::new(SamplerChain::new())),
        }
    }

    pub fn add(&self, sampler: Box<dyn Sampler>) {
        let mut chain = self.chain.lock().unwrap();
        chain.samplers.push(sampler);
    }

    pub fn sample(&self, logits: &mut [TokenData]) -> Result<i32> {
        let mut chain = self.chain.lock().unwrap();
        chain.sample(logits)
    }

    pub fn reset(&self) {
        let mut chain = self.chain.lock().unwrap();
        chain.reset();
    }
}

impl Clone for SharedSamplerChain {
    fn clone(&self) -> Self {
        Self {
            chain: Arc::clone(&self.chain),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temperature::Temperature;
    use crate::top_k::TopK;

    #[test]
    fn test_sampler_chain() {
        let mut chain = SamplerChain::new()
            .add(Box::new(TopK::new(40)))
            .add(Box::new(Temperature::new(0.8)));

        let mut logits = vec![
            TokenData::new(0, 1.0),
            TokenData::new(1, 2.0),
            TokenData::new(2, 0.5),
        ];

        let token = chain.sample(&mut logits);
        assert!(token.is_ok());
    }
}
