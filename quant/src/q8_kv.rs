//! Q8_KV quantization for KV cache
//!
//! 8-bit quantization specifically designed for KV cache in transformers
//! Provides good balance between memory savings and computational accuracy
//!
//! Unlike weight quantization, KV cache quantization:
//! - Uses simpler 8-bit linear quantization (no super-blocks)
//! - Per-tensor or per-block scales
//! - Focuses on maintaining accuracy for attention computation

use barq_core::error::{Error, Result};

pub const QK_KV: usize = 64;  // KV cache quantization block size

/// Q8_KV block for KV cache quantization
///
/// Simple 8-bit quantization: x = scale * q
/// Block size: QK_KV (64 elements)
/// 8 bytes per 64 values (effectively 8 bits per value)
///
/// Layout:
/// - qs[64]: quantized values (64 bytes)
/// - d: scale (f32, 4 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8KV {
    pub qs: [u8; QK_KV],
    pub d: f32,
}

impl BlockQ8KV {
    pub const fn size_bytes() -> usize {
        QK_KV + 4
    }

    pub fn dequantize(&self, output: &mut [f32]) {
        for (i, &q) in self.qs.iter().enumerate() {
            if i < output.len() {
                output[i] = self.d * (q as i32 - 128) as f32;
            }
        }
    }

    pub fn quantize(data: &[f32; QK_KV]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in data {
            max_abs = max_abs.max(v.abs());
        }

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

        let mut qs = [0u8; QK_KV];
        for (i, &v) in data.iter().enumerate() {
            let q = (v / scale).round().min(127.0).max(-127.0) as i8;
            qs[i] = (q + 128) as u8;
        }

        BlockQ8KV { qs, d: scale }
    }
}

pub struct Q8KV {
    block_size: usize,
}

impl Q8KV {
    pub fn new() -> Self {
        Self { block_size: QK_KV }
    }

    pub fn quantize(&self, input: &[f32]) -> Result<Vec<u8>> {
        let n_blocks = input.len() / QK_KV;
        let remainder = input.len() % QK_KV;

        let total_blocks = if remainder > 0 {
            n_blocks + 1
        } else {
            n_blocks
        };
        let mut output = Vec::with_capacity(total_blocks * BlockQ8KV::size_bytes());

        for block_idx in 0..n_blocks {
            let start = block_idx * QK_KV;
            let block: &[f32; QK_KV] = input[start..start + QK_KV]
                .try_into()
                .map_err(|_| Error::Quantization("Invalid block size".into()))?;

            let qblock = BlockQ8KV::quantize(block);
            output.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &qblock as *const BlockQ8KV as *const u8,
                    std::mem::size_of::<BlockQ8KV>(),
                )
            });
        }

        if remainder > 0 {
            let mut last_block = [0.0f32; QK_KV];
            last_block[..remainder].copy_from_slice(&input[n_blocks * QK_KV..]);
            let qblock = BlockQ8KV::quantize(&last_block);
            output.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &qblock as *const BlockQ8KV as *const u8,
                    std::mem::size_of::<BlockQ8KV>(),
                )
            });
        }

        Ok(output)
    }

    pub fn dequantize(&self, input: &[u8], output_size: usize) -> Result<Vec<f32>> {
        let block_bytes = BlockQ8KV::size_bytes();
        let n_blocks = (output_size + QK_KV - 1) / QK_KV;

        let mut output = vec![0.0f32; n_blocks * QK_KV];

        for (block_idx, out_chunk) in output.chunks_mut(QK_KV).enumerate().take(n_blocks) {
            let offset = block_idx * block_bytes;
            if offset + block_bytes > input.len() {
                break;
            }

            let qblock: &BlockQ8KV = unsafe { &*(input.as_ptr().add(offset) as *const BlockQ8KV) };

            qblock.dequantize(out_chunk);
        }

        output.truncate(output_size);
        Ok(output)
    }
}

impl Default for Q8KV {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_kv_block_size() {
        let q = Q8KV::new();
        assert_eq!(q.block_size, QK_KV);
    }

    #[test]
    fn test_q8_kv_size() {
        assert_eq!(BlockQ8KV::size_bytes(), 68);
    }

    #[test]
    fn test_q8_kv_roundtrip() {
        let quant = Q8KV::new();

        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());

        for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error < 0.1,
                "Error at index {}: {} vs {} (error={})",
                i,
                orig,
                deq,
                error
            );
        }
    }

    #[test]
    fn test_q8_kv_symmetric_values() {
        let quant = Q8KV::new();

        let input: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0]
            .into_iter()
            .cycle()
            .take(64)
            .collect();
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());

        for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error < 0.01,
                "Error at index {}: {} vs {} (error={})",
                i,
                orig,
                deq,
                error
            );
        }
    }
}
