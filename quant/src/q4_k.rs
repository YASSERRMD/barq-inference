//! Q4_K quantization implementation
//!
//! More sophisticated 4-bit quantization with separate scales for min/max

use core::tensor::{Tensor, TensorType, Shape};
use core::error::{Error, Result};

/// Q4_K quantization: 4-bit with min/max scaling
///
/// Layout: scale_min (f32), scale_max (f32), 4-bit quants (uint8 packed)
/// Block size: 256 weights
/// Bits per weight: 4.5

#[derive(Debug, Clone)]
pub struct Q4_K {
    block_size: usize,
}

impl Q4_K {
    pub fn new() -> Self {
        Self {
            block_size: 256,
        }
    }

    pub fn quantize(&self, input: &[f32]) -> Result<Vec<u8>> {
        let block_size = self.block_size;
        let n_blocks = (input.len() + block_size - 1) / block_size;

        let mut output = Vec::new();

        for block_idx in 0..n_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(input.len());
            let block = &input[start..end];

            if block.is_empty() {
                continue;
            }

            // Find min and max values
            let min_val = block.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
            let max_val = block.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

            // Compute scales
            let scale_min = min_val;
            let scale_max = max_val - min_val;

            // Quantize block to 4-bit
            let mut quants = vec![0u8; (block.len() + 1) / 2];

            for (i, &val) in block.iter().enumerate() {
                // Normalize to [0, 15] range
                let normalized = if scale_max == 0.0 {
                    7.5f32
                } else {
                    ((val - scale_min) / scale_max * 15.0)
                };

                let q = normalized.round().clamp(0.0, 15.0) as u8;

                // Pack two 4-bit values into one byte
                let byte_idx = i / 2;
                let shift = if i % 2 == 0 { 0 } else { 4 };
                quants[byte_idx] |= (q << shift);
            }

            // Output scale_min
            output.extend_from_slice(&scale_min.to_le_bytes());

            // Output scale_max
            output.extend_from_slice(&scale_max.to_le_bytes());

            // Output quantized values
            output.extend_from_slice(&quants);
        }

        Ok(output)
    }

    pub fn dequantize(&self, input: &[u8], output_size: usize) -> Result<Vec<f32>> {
        let block_size = self.block_size;
        let n_blocks = (output_size + block_size - 1) / block_size;

        let mut output = Vec::with_capacity(output_size);
        let mut offset = 0;

        for _ in 0..n_blocks {
            // Read scale_min
            if offset + 8 > input.len() {
                break;
            }

            let scale_min = f32::from_le_bytes([
                input[offset],
                input[offset + 1],
                input[offset + 2],
                input[offset + 3],
            ]);
            offset += 4;

            let scale_max = f32::from_le_bytes([
                input[offset],
                input[offset + 1],
                input[offset + 2],
                input[offset + 3],
            ]);
            offset += 4;

            // Read quantized values
            let q_len = (block_size + 1) / 2;
            if offset + q_len > input.len() {
                break;
            }

            let quants = &input[offset..offset + q_len];
            offset += q_len;

            // Dequantize block
            for i in 0..block_size {
                let byte_idx = i / 2;
                let shift = if i % 2 == 0 { 0 } else { 4 };

                if byte_idx < quants.len() {
                    let q = (quants[byte_idx] >> shift) & 0x0F;
                    let val = scale_min + (q as f32 / 15.0) * scale_max;
                    output.push(val);
                }
            }
        }

        output.truncate(output_size);
        Ok(output)
    }
}

impl Default for Q4_K {
    fn default() -> Self {
        Self::new()
    }
}

/// Q4_KS: Q4_K with super-blocks
#[derive(Debug, Clone)]
pub struct Q4_KS {
    block_size: usize,
    super_block_size: usize,
}

impl Q4_KS {
    pub fn new() -> Self {
        Self {
            block_size: 256,
            super_block_size: 1024,
        }
    }
}

impl Default for Q4_KS {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_k_roundtrip() {
        let quant = Q4_K::new();

        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());

        // Verify shape is preserved
        for i in 0..input.len() {
            assert!(!dequantized[i].is_nan());
        }
    }

    #[test]
    fn test_q4_k_block_size() {
        let quant = Q4_K::new();
        assert_eq!(quant.block_size, 256);
    }
}
