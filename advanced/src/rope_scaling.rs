//! RoPE (Rotary Position Embedding) scaling implementations
//!
//! Supports various scaling methods for extending context length:
//! - Linear scaling
//! - YaRN scaling
//! - NTK-aware scaling
//! - LongRoPE

use barq_core::error::Result;

/// RoPE scaling type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingType {
    /// No scaling
    None,
    /// Linear scaling
    Linear,
    /// YaRN scaling
    Yarn,
    /// NTK-aware scaling
    NtkAware,
    /// LongRoPE scaling
    LongRope,
}

/// RoPE scaling configuration
#[derive(Debug, Clone)]
pub struct RopeScalingConfig {
    /// Scaling type
    pub scaling_type: RopeScalingType,
    /// Original context length
    pub original_ctx_len: usize,
    /// Target context length
    pub target_ctx_len: usize,
    /// Base frequency
    pub freq_base: f32,
    /// Scale factor
    pub scale: f32,
}

impl Default for RopeScalingConfig {
    fn default() -> Self {
        Self {
            scaling_type: RopeScalingType::None,
            original_ctx_len: 2048,
            target_ctx_len: 2048,
            freq_base: 10000.0,
            scale: 1.0,
        }
    }
}

/// RoPE scaling trait
pub trait RopeScaling: Send + Sync {
    /// Apply scaling to positions
    fn apply(&self, positions: &[usize], dim: usize) -> Result<Vec<f32>>;

    /// Returns the scaling configuration
    fn config(&self) -> &RopeScalingConfig;
}

/// YaRN scaling implementation
pub struct YaRNScaling {
    config: RopeScalingConfig,
}

impl YaRNScaling {
    pub fn new(config: RopeScalingConfig) -> Self {
        Self { config }
    }

    fn compute_yarn_scale(&self, pos: usize) -> f32 {
        let ratio = self.config.target_ctx_len as f32 / self.config.original_ctx_len as f32;
        let progress = pos as f32 / self.config.target_ctx_len as f32;

        // YaRN scaling function
        if progress <= 0.1 {
            1.0 + (ratio - 1.0) * progress / 0.1
        } else if progress <= 0.9 {
            ratio
        } else {
            ratio - (ratio - 1.0) * (progress - 0.9) / 0.1
        }
    }
}

impl RopeScaling for YaRNScaling {
    fn apply(&self, positions: &[usize], dim: usize) -> Result<Vec<f32>> {
        let mut scaled = Vec::with_capacity(positions.len() * dim);

        for &pos in positions {
            let scale = self.compute_yarn_scale(pos);

            for i in 0..dim {
                let theta = (pos as f32) / self.config.freq_base.powf(2.0 * i as f32 / dim as f32);
                scaled.push(theta * scale);
            }
        }

        Ok(scaled)
    }

    fn config(&self) -> &RopeScalingConfig {
        &self.config
    }
}

/// NTK-aware scaling
pub struct NtkAwareScaling {
    config: RopeScalingConfig,
}

impl NtkAwareScaling {
    pub fn new(config: RopeScalingConfig) -> Self {
        Self { config }
    }
}

impl RopeScaling for NtkAwareScaling {
    fn apply(&self, positions: &[usize], dim: usize) -> Result<Vec<f32>> {
        let alpha = (self.config.target_ctx_len as f32 / self.config.original_ctx_len as f32)
            .powf(dim as f32 / (dim as f32 - 2.0));

        let mut scaled = Vec::with_capacity(positions.len() * dim);

        for &pos in positions {
            for i in 0..dim {
                let theta = (pos as f32)
                    / (self.config.freq_base * alpha).powf(2.0 * i as f32 / dim as f32);
                scaled.push(theta);
            }
        }

        Ok(scaled)
    }

    fn config(&self) -> &RopeScalingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yarn_scaling() {
        let config = RopeScalingConfig {
            scaling_type: RopeScalingType::Yarn,
            original_ctx_len: 2048,
            target_ctx_len: 4096,
            ..Default::default()
        };

        let yarn = YaRNScaling::new(config);
        let scale = yarn.compute_yarn_scale(1024);

        assert!(scale > 1.0 && scale <= 2.0);
    }

    #[test]
    fn test_rope_config() {
        let config = RopeScalingConfig::default();
        assert_eq!(config.scaling_type, RopeScalingType::None);
        assert_eq!(config.original_ctx_len, 2048);
    }
}
