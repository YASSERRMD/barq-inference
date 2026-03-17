//! Q4_0 quantization implementation
//!
//! 4-bit quantization with scale per block

use barq_core::tensor::{Tensor, TensorType, Shape};
use barq_core::error::{Error, Result};

/// Q4_0 quantization: 4-bit weights with per-block scale
///
/// Layout: scale (f32), 4-bit quants (uint8 packed)
/// Block size: 32 weights
/// Bits per weight: 4.5

#[derive(Debug, Clone)]
pub struct Q4_0 {
    block_size: usize,
}

impl Q4_0 {
    pub fn new() -> Self {
        Self {
            block_size: 32,
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

            // Find max absolute value for scaling
            let max_abs = block.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

            // Compute scale: max_abs / -8.0 (since 4-bit signed range is -8 to 7)
            let scale = if max_abs == 0.0 {
                0.0
            } else {
                max_abs / 8.0
            };

            // Quantize block
            let mut quants = vec![0u8; (block.len() + 1) / 2];

            for (i, &val) in block.iter().enumerate() {
                let q = if scale == 0.0 {
                    0i8
                } else {
                    (val / scale).round() as i8
                };

                // Clamp to valid range
                let q = q.max(-8).min(7);

                // Pack two 4-bit values into one byte
                let byte_idx = i / 2;
                let shift = if i % 2 == 0 { 0 } else { 4 };
                quants[byte_idx] |= ((q as u8 & 0x0F) << shift);
            }

            // Output scale (as f32 bytes)
            output.extend_from_slice(&scale.to_le_bytes());

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
            // Read scale
            if offset + 4 > input.len() {
                break;
            }

            let scale = f32::from_le_bytes([
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
                    let q = ((quants[byte_idx] >> shift) & 0x0F) as i8;
                    let q = if q >= 8 { q - 16 } else { q }; // Convert to signed
                    output.push(q as f32 * scale);
                }
            }
        }

        output.truncate(output_size);
        Ok(output)
    }
}

impl Default for Q4_0 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        let quant = Q4_0::new();

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());

        // Check values are approximately correct (within quantization error)
        for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(error < 0.5, "Error at index {}: {} vs {}", i, orig, deq);
        }
    }
}
