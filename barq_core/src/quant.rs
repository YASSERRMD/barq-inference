//! Quantization types and utilities

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization (F32)
    F32,
    /// 16-bit floating point
    F16,
    /// Block-wise 4-bit quantization (0)
    Q4_0,
    /// Block-wise 4-bit quantization (1)
    Q4_1,
    /// Block-wise 5-bit quantization (0)
    Q5_0,
    /// Block-wise 5-bit quantization (1)
    Q5_1,
    /// Block-wise 8-bit quantization (0)
    Q8_0,
    /// K-quants
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    /// I-quants
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    IQ1_M,
    /// T-quants
    TQ1_0,
    TQ2_0,
}

impl QuantizationType {
    /// Returns the block size for this quantization type
    pub const fn block_size(&self) -> usize {
        match self {
            QuantizationType::Q4_0 => 32,
            QuantizationType::Q4_1 => 32,
            QuantizationType::Q5_0 => 32,
            QuantizationType::Q5_1 => 32,
            QuantizationType::Q8_0 => 32,
            QuantizationType::Q2_K => 256,
            QuantizationType::Q3_K => 256,
            QuantizationType::Q4_K => 256,
            QuantizationType::Q5_K => 256,
            QuantizationType::Q6_K => 256,
            QuantizationType::Q8_K => 256,
            QuantizationType::IQ2_XXS => 256,
            QuantizationType::IQ2_XS => 256,
            QuantizationType::IQ3_XXS => 256,
            QuantizationType::IQ1_S => 256,
            QuantizationType::IQ4_NL => 32,
            QuantizationType::IQ3_S => 256,
            QuantizationType::IQ2_S => 256,
            QuantizationType::IQ4_XS => 256,
            QuantizationType::IQ1_M => 256,
            QuantizationType::TQ1_0 => 256,
            QuantizationType::TQ2_0 => 256,
            QuantizationType::F32 | QuantizationType::F16 => 1,
        }
    }

    /// Returns the bits per value for this quantization type
    pub const fn bits(&self) -> f32 {
        match self {
            QuantizationType::F32 => 32.0,
            QuantizationType::F16 => 16.0,
            QuantizationType::Q4_0 | QuantizationType::Q4_1 => 4.5,
            QuantizationType::Q5_0 | QuantizationType::Q5_1 => 5.5,
            QuantizationType::Q8_0 => 8.5,
            QuantizationType::Q2_K => 2.5625,
            QuantizationType::Q3_K => 3.4375,
            QuantizationType::Q4_K => 4.5,
            QuantizationType::Q5_K => 5.5,
            QuantizationType::Q6_K => 6.5625,
            QuantizationType::Q8_K => 8.5,
            QuantizationType::IQ2_XXS => 2.0625,
            QuantizationType::IQ2_XS => 2.5,
            QuantizationType::IQ3_XXS => 3.0625,
            QuantizationType::IQ1_S => 1.5625,
            QuantizationType::IQ4_NL => 4.5,
            QuantizationType::IQ3_S => 3.4375,
            QuantizationType::IQ2_S => 2.5,
            QuantizationType::IQ4_XS => 4.5,
            QuantizationType::IQ1_M => 1.75,
            QuantizationType::TQ1_0 => 2.0625,
            QuantizationType::TQ2_0 => 3.0625,
        }
    }

    /// Returns the type name
    pub fn name(&self) -> &'static str {
        match self {
            QuantizationType::F32 => "f32",
            QuantizationType::F16 => "f16",
            QuantizationType::Q4_0 => "q4_0",
            QuantizationType::Q4_1 => "q4_1",
            QuantizationType::Q5_0 => "q5_0",
            QuantizationType::Q5_1 => "q5_1",
            QuantizationType::Q8_0 => "q8_0",
            QuantizationType::Q2_K => "q2_k",
            QuantizationType::Q3_K => "q3_k",
            QuantizationType::Q4_K => "q4_k",
            QuantizationType::Q5_K => "q5_k",
            QuantizationType::Q6_K => "q6_k",
            QuantizationType::Q8_K => "q8_k",
            QuantizationType::IQ2_XXS => "iq2_xxs",
            QuantizationType::IQ2_XS => "iq2_xs",
            QuantizationType::IQ3_XXS => "iq3_xxs",
            QuantizationType::IQ1_S => "iq1_s",
            QuantizationType::IQ4_NL => "iq4_nl",
            QuantizationType::IQ3_S => "iq3_s",
            QuantizationType::IQ2_S => "iq2_s",
            QuantizationType::IQ4_XS => "iq4_xs",
            QuantizationType::IQ1_M => "iq1_m",
            QuantizationType::TQ1_0 => "tq1_0",
            QuantizationType::TQ2_0 => "tq2_0",
        }
    }

    /// Parse quantization type from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" => Ok(QuantizationType::F32),
            "f16" | "float16" => Ok(QuantizationType::F16),
            "q4_0" => Ok(QuantizationType::Q4_0),
            "q4_1" => Ok(QuantizationType::Q4_1),
            "q5_0" => Ok(QuantizationType::Q5_0),
            "q5_1" => Ok(QuantizationType::Q5_1),
            "q8_0" => Ok(QuantizationType::Q8_0),
            "q2_k" => Ok(QuantizationType::Q2_K),
            "q3_k" => Ok(QuantizationType::Q3_K),
            "q4_k" => Ok(QuantizationType::Q4_K),
            "q5_k" => Ok(QuantizationType::Q5_K),
            "q6_k" => Ok(QuantizationType::Q6_K),
            "q8_k" => Ok(QuantizationType::Q8_K),
            "iq2_xxs" => Ok(QuantizationType::IQ2_XXS),
            "iq2_xs" => Ok(QuantizationType::IQ2_XS),
            "iq3_xxs" => Ok(QuantizationType::IQ3_XXS),
            "iq1_s" => Ok(QuantizationType::IQ1_S),
            "iq4_nl" => Ok(QuantizationType::IQ4_NL),
            "iq3_s" => Ok(QuantizationType::IQ3_S),
            "iq2_s" => Ok(QuantizationType::IQ2_S),
            "iq4_xs" => Ok(QuantizationType::IQ4_XS),
            "iq1_m" => Ok(QuantizationType::IQ1_M),
            "tq1_0" => Ok(QuantizationType::TQ1_0),
            "tq2_0" => Ok(QuantizationType::TQ2_0),
            _ => Err(Error::Quantization(format!(
                "Unknown quantization type: {}",
                s
            ))),
        }
    }
}

/// Quantization trait
pub trait Quantize {
    /// Quantize the data
    fn quantize(&self, qtype: QuantizationType) -> Result<Vec<u8>>;
}

/// Dequantization trait
pub trait Dequantize {
    /// Dequantize the data
    fn dequantize(&self, qtype: QuantizationType, data: &[u8]) -> Result<Vec<f32>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type() {
        let qtype = QuantizationType::Q4_K;
        assert_eq!(qtype.block_size(), 256);
        assert_eq!(qtype.bits(), 4.5);
        assert_eq!(qtype.name(), "q4_k");
    }

    #[test]
    fn test_parse_quantization_type() {
        assert_eq!(
            QuantizationType::from_str("q4_k").unwrap(),
            QuantizationType::Q4_K
        );
        assert_eq!(
            QuantizationType::from_str("Q4_K").unwrap(),
            QuantizationType::Q4_K
        );
        assert_eq!(
            QuantizationType::from_str("f32").unwrap(),
            QuantizationType::F32
        );
        assert!(QuantizationType::from_str("unknown").is_err());
    }
}
