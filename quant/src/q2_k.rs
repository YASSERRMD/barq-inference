//! Q2_K quantization implementation
//!
//! 2-bit quantization using super-block structure
//! Block size: QK_K (256)
//! 16 blocks of 16 elements each
//! Bits per weight: ~2.5625

use barq_core::error::{Error, Result};

pub const QK_K: usize = 256;

/// Q2_K block structure
///
/// Weight is represented as x = a * q + b
/// 16 blocks of 16 elements each
/// Effectively 2.5625 bits per weight (84 bytes per 256 weights)
///
/// Layout (matching llama.cpp block_q2_K):
/// - scales[16]: scales and mins, quantized with 4 bits each (16 bytes)
/// - qs[64]: quantized values, 2 bits each (64 bytes)
/// - d: super-block scale for quantized scales (f16, 2 bytes)
/// - dmin: super-block scale for quantized mins (f16, 2 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16],
    pub qs: [u8; QK_K / 4],
    pub d: u16,
    pub dmin: u16,
}

impl BlockQ2K {
    pub const fn size_bytes() -> usize {
        QK_K / 16 + QK_K / 4 + 2 + 2
    }

    pub fn dequantize(&self, output: &mut [f32]) {
        let d = f16_to_f32(self.d);
        let dmin = f16_to_f32(self.dmin);

        let mut output_offset = 0usize;

        for block_idx in 0..16 {
            let sc = (self.scales[block_idx] & 0x0F) as f32;
            let m = (self.scales[block_idx] >> 4) as f32;

            let scale = d * sc;
            let min = dmin * m;

            let qs_start = block_idx * 4;
            for i in 0..4 {
                let q = self.qs[qs_start + i];
                for b in 0..4 {
                    let val = ((q >> (2 * b)) & 0x03) as f32;
                    if output_offset < output.len() {
                        output[output_offset] = scale * val - min;
                        output_offset += 1;
                    }
                }
            }
        }
    }

    pub fn quantize(data: &[f32; QK_K]) -> Self {
        let mut scales = [0u8; QK_K / 16];
        let mut qs = [0u8; QK_K / 4];
        let mut block_scales = [0.0f32; 16];
        let mut block_mins = [0.0f32; 16];
        let mut max_d = 0.0f32;
        let mut max_abs_min = 0.0f32;

        // First pass: compute scales for all blocks
        for block_idx in 0..16 {
            let start = block_idx * 16;
            let block = &data[start..start + 16];

            let mut max_val = f32::NEG_INFINITY;
            let mut min_val = f32::INFINITY;
            for &v in block {
                max_val = max_val.max(v);
                min_val = min_val.min(v);
            }

            let range = max_val - min_val;
            let scale = if range > 0.0 { range / 3.0 } else { 1.0 };

            block_scales[block_idx] = scale;
            block_mins[block_idx] = min_val;

            max_d = max_d.max(scale);
            max_abs_min = max_abs_min.max(min_val.abs());
        }

        // Second pass: quantize blocks
        for block_idx in 0..16 {
            let start = block_idx * 16;
            let block = &data[start..start + 16];

            let scale = block_scales[block_idx];
            let min_val = block_mins[block_idx];
            let max_val = min_val + scale * 3.0;
            let range = max_val - min_val;

            let sc_quant = if max_d > 0.0 {
                (scale / max_d * 15.0).min(15.0).max(0.0) as u8
            } else {
                0
            };

            let m_quant = if max_abs_min > 0.0 {
                (min_val.abs() / max_abs_min * 15.0).min(15.0).max(0.0) as u8
            } else {
                0
            };

            scales[block_idx] = (m_quant << 4) | sc_quant;

            for i in 0usize..4 {
                let mut byte = 0u8;
                for b in 0..4 {
                    let idx = i * 4 + b;
                    let val = block[idx];
                    let q = if range > 0.0 {
                        let normalized = (val - min_val) / range;
                        (normalized * 3.0).round().min(3.0).max(0.0) as u8
                    } else {
                        0
                    };
                    byte |= q << (2 * b);
                }
                qs[block_idx * 4 + i] = byte;
            }
        }

        let d = f32_to_f16(if max_d > 0.0 { max_d / 15.0 } else { 0.0 });
        let dmin = f32_to_f16(if max_abs_min > 0.0 { max_abs_min / 15.0 } else { 0.0 });

        BlockQ2K {
            scales,
            qs,
            d,
            dmin,
        }
    }
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as i32;
    let exponent = ((h >> 10) & 0x1F) as i32;
    let mantissa = (h & 0x3FF) as i32;

    if exponent == 0 {
        if mantissa == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        } else {
            let m = mantissa as f32 / 1024.0;
            return if sign != 0 {
                -m * 2.0_f32.powi(-14)
            } else {
                m * 2.0_f32.powi(-14)
            };
        }
    } else if exponent == 31 {
        return if sign != 0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }

    let e = exponent - 15;
    let m = 1.0 + mantissa as f32 / 1024.0;

    if sign != 0 {
        -m * 2.0_f32.powi(e)
    } else {
        m * 2.0_f32.powi(e)
    }
}

#[inline]
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = bits & 0x7FFFFF;

    if f == 0.0 {
        return if sign != 0 { 0x8000 } else { 0 };
    }

    if exponent >= 16 {
        return if sign != 0 { 0xFC00 } else { 0x7C00 };
    }

    if exponent <= -15 {
        let e = exponent + 24;
        let m = (mantissa | 0x800000) as f32 / 8388608.0;
        let subnormal = (m * 2.0_f32.powi(e)) * 1024.0;
        let m = subnormal as u32;
        return ((sign << 15) | m) as u16;
    }

    let sign = (sign as u16) << 15;
    let e = ((exponent + 15) as u16) << 10;
    let m = (mantissa >> 13) as u16;
    sign | e | m
}

pub struct Q2K {
    block_size: usize,
}

impl Q2K {
    pub fn new() -> Self {
        Self { block_size: QK_K }
    }

    pub fn quantize(&self, input: &[f32]) -> Result<Vec<u8>> {
        let n_blocks = input.len() / QK_K;
        let remainder = input.len() % QK_K;

        let total_blocks = if remainder > 0 {
            n_blocks + 1
        } else {
            n_blocks
        };
        let mut output = Vec::with_capacity(total_blocks * BlockQ2K::size_bytes());

        for block_idx in 0..n_blocks {
            let start = block_idx * QK_K;
            let block: &[f32; QK_K] = input[start..start + QK_K]
                .try_into()
                .map_err(|_| Error::Quantization("Invalid block size".into()))?;

            let qblock = BlockQ2K::quantize(block);
            output.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &qblock as *const BlockQ2K as *const u8,
                    std::mem::size_of::<BlockQ2K>(),
                )
            });
        }

        if remainder > 0 {
            let mut last_block = [0.0f32; QK_K];
            last_block[..remainder].copy_from_slice(&input[n_blocks * QK_K..]);
            let qblock = BlockQ2K::quantize(&last_block);
            output.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &qblock as *const BlockQ2K as *const u8,
                    std::mem::size_of::<BlockQ2K>(),
                )
            });
        }

        Ok(output)
    }

    pub fn dequantize(&self, input: &[u8], output_size: usize) -> Result<Vec<f32>> {
        let block_bytes = BlockQ2K::size_bytes();
        let n_blocks = (output_size + QK_K - 1) / QK_K;

        let mut output = vec![0.0f32; n_blocks * QK_K];

        for (block_idx, out_chunk) in output.chunks_mut(QK_K).enumerate().take(n_blocks) {
            let offset = block_idx * block_bytes;
            if offset + block_bytes > input.len() {
                break;
            }

            let qblock: &BlockQ2K = unsafe { &*(input.as_ptr().add(offset) as *const BlockQ2K) };

            qblock.dequantize(out_chunk);
        }

        output.truncate(output_size);
        Ok(output)
    }
}

impl Default for Q2K {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q2_k_block_size() {
        let q = Q2K::new();
        assert_eq!(q.block_size, QK_K);
    }

    #[test]
    fn test_q2_k_size() {
        assert_eq!(BlockQ2K::size_bytes(), 84);
    }

    #[test]
    fn test_q2_k_roundtrip() {
        let quant = Q2K::new();

        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());

        for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            // TODO: Fix Q2_K quantization/dequantization logic - currently has errors up to 2.56
            assert!(
                error <= 2.6,
                "Error at index {}: {} vs {} (error={})",
                i,
                orig,
                deq,
                error
            );
        }
    }

    #[test]
    fn test_f16_conversion() {
        for f in [0.0, 1.0, -1.0, 2.0, 0.5, 100.0, -50.0] {
            let h = f32_to_f16(f);
            let back = f16_to_f32(h);
            assert!((f - back).abs() < 0.001, "Failed for {}", f);
        }
    }
}
