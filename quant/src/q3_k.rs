//! Q3_K quantization implementation
//!
//! 3-bit quantization using super-block structure
//! Block size: QK_K (256)
//! Effectively 3.4375 bits per weight

use barq_core::error::{Error, Result};

pub const QK_K: usize = 256;

/// Q3_K block structure
///
/// Weight is represented as x = a * q
/// 16 blocks of 16 elements each
/// Effectively 3.4375 bits per weight (110 bytes per 256 weights)
///
/// Layout (matching llama.cpp block_q3_K):
/// - hmask[32]: high bit mask for quantized values (32 bytes)
/// - qs[64]: quantized values, 2 bits each (64 bytes)
/// - scales[12]: scales, quantized with 6 bits (12 bytes)
/// - d: super-block scale (f16, 2 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; 12],
    pub d: u16,
}

impl BlockQ3K {
    pub const fn size_bytes() -> usize {
        QK_K / 8 + QK_K / 4 + 12 + 2
    }

    pub fn dequantize(&self, output: &mut [f32]) {
        let d = f16_to_f32(self.d);

        let mut output_offset = 0usize;
        let mut scale_idx = 0usize;

        for block_idx in 0..16 {
            let sc = get_scale_q3k(scale_idx, &self.scales);
            scale_idx += 1;

            let scale = d * sc;

            let hmask_start = block_idx * 2;
            let qs_start = block_idx * 4;

            for i in 0..4 {
                let q = self.qs[qs_start + i];
                let h = self.hmask[hmask_start + (i / 2)];
                let hbit = if i % 2 == 0 {
                    (h & 0x0F, 0)
                } else {
                    (h >> 4, 4)
                };

                for b in 0..4 {
                    let low_bits = ((q >> (2 * b)) & 0x03) as i8;
                    let high_bit = if (hbit.0 >> b) & 1 != 0 { 4i8 } else { 0i8 };
                    let val = low_bits | high_bit;
                    let signed = val - 4;

                    if output_offset < output.len() {
                        output[output_offset] = scale * (signed as f32);
                        output_offset += 1;
                    }
                }
            }
        }
    }

    pub fn quantize(data: &[f32; QK_K]) -> Self {
        let mut hmask = [0u8; QK_K / 8];
        let mut qs = [0u8; QK_K / 4];
        let mut scales = [0u8; 12];

        let mut all_max_abs = 0.0f32;

        for block_idx in 0..16 {
            let start = block_idx * 16;
            let block = &data[start..start + 16];

            let mut max_abs = 0.0f32;
            for &v in block {
                max_abs = max_abs.max(v.abs());
            }
            all_max_abs = all_max_abs.max(max_abs);

            let scale = if max_abs > 0.0 { max_abs / 4.0 } else { 0.0 };

            let scale_quant = if scale > 0.0 {
                ((max_abs / all_max_abs).min(1.0) * 31.0).min(31.0) as u8
            } else {
                0
            };

            if block_idx < 12 {
                scales[block_idx] = scale_quant;
            }

            for i in 0..4 {
                let mut byte = 0u8;
                let mut hbyte = 0u8;

                for b in 0..4 {
                    let idx = i * 4 + b;
                    let val = block[idx];
                    let normalized = if scale > 0.0 { val / scale } else { 0.0 };

                    let q_signed = (normalized * 4.0).round().max(-4.0).min(3.0) as i8;
                    let q_positive = (q_signed + 4) as u8;

                    let low_bits = q_positive & 0x03;
                    let high_bit = (q_positive >> 2) & 0x01;

                    byte |= low_bits << (2 * b);
                    hbyte |= high_bit << b;
                }

                qs[block_idx * 4 + i] = byte;
                if i % 2 == 0 {
                    hmask[block_idx * 2 + (i / 2)] = hbyte;
                } else {
                    hmask[block_idx * 2 + (i / 2)] |= hbyte << 4;
                }
            }
        }

        let d = f32_to_f16(all_max_abs / 31.0);

        BlockQ3K {
            hmask,
            qs,
            scales,
            d,
        }
    }
}

#[inline]
fn get_scale_q3k(j: usize, scales: &[u8]) -> f32 {
    if j < 4 {
        (scales[j] & 0x3F) as f32
    } else {
        let tmp = scales[j - 4];
        ((tmp & 0x0F) | ((scales[j] & 0x0F) << 4)) as f32
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

pub struct Q3K {
    block_size: usize,
}

impl Q3K {
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
        let mut output = Vec::with_capacity(total_blocks * BlockQ3K::size_bytes());

        for block_idx in 0..n_blocks {
            let start = block_idx * QK_K;
            let block: &[f32; QK_K] = input[start..start + QK_K]
                .try_into()
                .map_err(|_| Error::Quantization("Invalid block size".into()))?;

            let qblock = BlockQ3K::quantize(block);
            output.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &qblock as *const BlockQ3K as *const u8,
                    std::mem::size_of::<BlockQ3K>(),
                )
            });
        }

        if remainder > 0 {
            let mut last_block = [0.0f32; QK_K];
            last_block[..remainder].copy_from_slice(&input[n_blocks * QK_K..]);
            let qblock = BlockQ3K::quantize(&last_block);
            output.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &qblock as *const BlockQ3K as *const u8,
                    std::mem::size_of::<BlockQ3K>(),
                )
            });
        }

        Ok(output)
    }

    pub fn dequantize(&self, input: &[u8], output_size: usize) -> Result<Vec<f32>> {
        let block_bytes = BlockQ3K::size_bytes();
        let n_blocks = (output_size + QK_K - 1) / QK_K;

        let mut output = vec![0.0f32; n_blocks * QK_K];

        for (block_idx, out_chunk) in output.chunks_mut(QK_K).enumerate().take(n_blocks) {
            let offset = block_idx * block_bytes;
            if offset + block_bytes > input.len() {
                break;
            }

            let qblock: &BlockQ3K = unsafe { &*(input.as_ptr().add(offset) as *const BlockQ3K) };

            qblock.dequantize(out_chunk);
        }

        output.truncate(output_size);
        Ok(output)
    }
}

impl Default for Q3K {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q3_k_block_size() {
        let q = Q3K::new();
        assert_eq!(q.block_size, QK_K);
    }

    #[test]
    fn test_q3_k_size() {
        assert_eq!(BlockQ3K::size_bytes(), 110);
    }

    #[test]
    fn test_q3_k_roundtrip() {
        let quant = Q3K::new();

        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 50.0).collect();
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());

        for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error < 0.2,
                "Error at index {}: {} vs {} (error={})",
                i,
                orig,
                deq,
                error
            );
        }
    }

    #[test]
    fn test_q3_k_signed_values() {
        let quant = Q3K::new();

        let input: Vec<f32> = vec![-4.0, -2.0, 0.0, 2.0, 4.0]
            .into_iter()
            .cycle()
            .take(256)
            .collect();
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert!(
            dequantized.iter().any(|&v| v < 0.0),
            "Should have negative values"
        );
    }
}
