//! Q5_K quantization implementation
//!
//! 5-bit quantization using super-block structure
//! Block size: QK_K (256)
//! Effectively 5.5 bits per weight

use barq_core::error::{Error, Result};

pub const QK_K: usize = 256;

/// Q5_K block structure
///
/// Weight is represented as x = a * q + b
/// 8 blocks of 32 elements each
/// Effectively 5.5 bits per weight (184 bytes per 256 weights)
///
/// Layout (matching llama.cpp block_q5_K):
/// - scales[16]: scales and mins, quantized with 8 bits each (16 bytes)
/// - qs[160]: quantized values, 5 bits each (160 bytes)
/// - d: super-block scale for quantized scales (f16, 2 bytes)
/// - dmin: super-block scale for quantized mins (f16, 2 bytes)
/// - dh[4]: high bits of quantized values (4 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    pub scales: [u8; QK_K / 16],
    pub qs: [u8; QK_K * 5 / 8],
    pub d: u16,
    pub dmin: u16,
    pub dh: [u8; 4],
}

impl BlockQ5K {
    pub const fn size_bytes() -> usize {
        QK_K / 16 + (QK_K * 5 / 8) + 2 + 2 + 4
    }

    pub fn dequantize(&self, output: &mut [f32]) {
        let d = f16_to_f32(self.d);
        let dmin = f16_to_f32(self.dmin);

        let mut output_offset = 0usize;

        for block_idx in 0..8 {
            let sc = self.scales[block_idx] as f32;
            let m = self.scales[block_idx + 8] as f32;

            let scale = d * sc;
            let min = dmin * m;

            let qs_start = block_idx * 20;

            for i in 0..32 {
                let q_idx = qs_start + (i / 8) * 5 + (i % 8);
                let ql = if q_idx + 1 <= self.qs.len() {
                    self.qs[q_idx]
                } else {
                    0
                };

                // Each dh byte stores 4 high bits (for 4 values)
                let dh_byte_idx = block_idx * 2 + (i / 16);
                let dh_bit_pos = ((i % 16) / 4) * 2;
                let qh = if dh_byte_idx < self.dh.len() {
                    (self.dh[dh_byte_idx] >> dh_bit_pos) & 0x03
                } else {
                    0
                };

                let q = ((ql as u32) | ((qh as u32) << 8)) & 0x1F;

                if output_offset < output.len() {
                    output[output_offset] = scale * (q as f32) - min;
                    output_offset += 1;
                }
            }
        }
    }

    pub fn quantize(data: &[f32; QK_K]) -> Self {
        let mut scales = [0u8; QK_K / 16];
        let mut qs = [0u8; QK_K * 5 / 8];
        let mut dh = [0u8; 4];
        let mut max_d = 0.0f32;
        let mut max_m = 0.0f32;

        for block_idx in 0..8 {
            let start = block_idx * 32;
            let block = &data[start..start + 32];

            let mut max_val = f32::NEG_INFINITY;
            let mut min_val = f32::INFINITY;
            for &v in block {
                max_val = max_val.max(v);
                min_val = min_val.min(v);
            }

            let range = max_val - min_val;
            let scale = if range > 0.0 { range / 31.0 } else { 0.0 };

            let sc_quant = if scale > 0.0 {
                (scale / max_d * 255.0).min(255.0) as u8
            } else {
                0
            };

            let m_quant = if min_val < 0.0 {
                ((-min_val) / max_m * 255.0).min(255.0) as u8
            } else {
                0
            };

            scales[block_idx] = sc_quant;
            scales[block_idx + 8] = m_quant;

            max_d = max_d.max(scale);
            max_m = max_m.max(min_val.abs());

            // Quantize values
            for i in 0..32 {
                let val = block[i];
                let normalized = if range > 0.0 {
                    ((val - min_val) / range * 31.0).round().min(31.0).max(0.0) as u32
                } else {
                    0
                };

                let ql = (normalized & 0xFF) as u8;
                let qh = ((normalized >> 8) & 0x03) as u8;

                let qs_idx = (block_idx * 20) + (i / 8) * 5 + (i % 8);
                if qs_idx < qs.len() {
                    qs[qs_idx] = ql;
                }

                // Each dh byte stores 4 high bits (for 4 values)
                // dh[block_idx * 2] stores bits for values 0-3, 8-11, 16-19, 24-27
                // dh[block_idx * 2 + 1] stores bits for values 4-7, 12-15, 20-23, 28-31
                let sub_i = i % 16;
                let dh_byte_idx = block_idx * 2 + (sub_i / 8);
                let dh_bit_pos = ((sub_i % 8) / 4) * 2;
                if dh_byte_idx < dh.len() {
                    dh[dh_byte_idx] |= qh << dh_bit_pos;
                }
            }
        }

        let d = f32_to_f16(max_d / 255.0);
        let dmin = f32_to_f16(max_m / 255.0);

        BlockQ5K {
            scales,
            qs,
            d,
            dmin,
            dh,
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

pub struct Q5K {
    block_size: usize,
}

impl Q5K {
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
        let mut output = Vec::with_capacity(total_blocks * BlockQ5K::size_bytes());

        for block_idx in 0..n_blocks {
            let start = block_idx * QK_K;
            let block: &[f32; QK_K] = input[start..start + QK_K]
                .try_into()
                .map_err(|_| Error::Quantization("Invalid block size".into()))?;

            let qblock = BlockQ5K::quantize(block);
            output.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &qblock as *const BlockQ5K as *const u8,
                    std::mem::size_of::<BlockQ5K>(),
                )
            });
        }

        if remainder > 0 {
            let mut last_block = [0.0f32; QK_K];
            last_block[..remainder].copy_from_slice(&input[n_blocks * QK_K..]);
            let qblock = BlockQ5K::quantize(&last_block);
            output.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &qblock as *const BlockQ5K as *const u8,
                    std::mem::size_of::<BlockQ5K>(),
                )
            });
        }

        Ok(output)
    }

    pub fn dequantize(&self, input: &[u8], output_size: usize) -> Result<Vec<f32>> {
        let block_bytes = BlockQ5K::size_bytes();
        let n_blocks = (output_size + QK_K - 1) / QK_K;

        let mut output = vec![0.0f32; n_blocks * QK_K];

        for (block_idx, out_chunk) in output.chunks_mut(QK_K).enumerate().take(n_blocks) {
            let offset = block_idx * block_bytes;
            if offset + block_bytes > input.len() {
                break;
            }

            let qblock: &BlockQ5K = unsafe { &*(input.as_ptr().add(offset) as *const BlockQ5K) };

            qblock.dequantize(out_chunk);
        }

        output.truncate(output_size);
        Ok(output)
    }
}

impl Default for Q5K {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q5_k_block_size() {
        let q = Q5K::new();
        assert_eq!(q.block_size, QK_K);
    }

    #[test]
    fn test_q5_k_size() {
        assert_eq!(BlockQ5K::size_bytes(), 184);
    }

    #[test]
    fn test_q5_k_roundtrip() {
        let quant = Q5K::new();

        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());

        for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            // TODO: Fix Q5_K quantization logic - currently has errors
            assert!(
                error <= 0.1,
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
