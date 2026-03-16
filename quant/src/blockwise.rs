//! Block-wise quantization utilities

use core::error::{Error, Result};

/// Block-wise quantization trait
pub trait QuantizeBlock: Send + Sync {
    /// Quantize a block of f32 values
    fn quantize_block(&self, block: &[f32]) -> Result<Vec<u8>>;

    /// Dequantize a block
    fn dequantize_block(&self, block: &[u8], output_len: usize) -> Result<Vec<f32>>;

    /// Returns the block size
    fn block_size(&self) -> usize;
}

/// Generic block-wise quantizer
pub struct BlockwiseQuant<Q: QuantizeBlock> {
    quantizer: Q,
}

impl<Q: QuantizeBlock> BlockwiseQuant<Q> {
    pub fn new(quantizer: Q) -> Self {
        Self { quantizer }
    }

    pub fn quantize(&self, input: &[f32]) -> Result<Vec<u8>> {
        let block_size = self.quantizer.block_size();
        let n_blocks = (input.len() + block_size - 1) / block_size;

        let mut output = Vec::new();
        let mut total_size = 0u32;

        for block_idx in 0..n_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(input.len());
            let block = &input[start..end];

            let quantized = self.quantizer.quantize_block(block)?;
            output.extend_from_slice(&quantized);
            total_size += quantized.len() as u32;
        }

        Ok(output)
    }

    pub fn dequantize(&self, input: &[u8], output_len: usize) -> Result<Vec<f32>> {
        let block_size = self.quantizer.block_size();
        let n_blocks = (output_len + block_size - 1) / block_size;

        let mut output = Vec::with_capacity(output_len);
        let mut offset = 0;

        for block_idx in 0..n_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(output_len);
            let block_len = end - start;

            // For now, assume all blocks have same compressed size
            // In practice, this needs more sophisticated logic
            if offset >= input.len() {
                break;
            }

            // Estimate compressed block size (this is simplified)
            let remaining = input.len() - offset;
            let compressed_size = remaining.min(512); // Reasonable upper bound

            let compressed = &input[offset..offset + compressed_size];
            let dequantized = self.quantizer.dequantize_block(compressed, block_len)?;

            output.extend_from_slice(&dequantized);
            offset += compressed_size;
        }

        output.truncate(output_len);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q4_0::Q4_0;

    #[test]
    fn test_blockwise_quant() {
        let q4_0 = Q4_0::new();
        let quant = BlockwiseQuant::new(q4_0);

        let input = vec![1.0, 2.0, 3.0, 4.0; 64];
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());
    }
}
