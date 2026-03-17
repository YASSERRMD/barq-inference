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
        let mut max_d = 0.0f32;
        let mut max_m = 0.0f32;

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
            let scale = if range > 0.0 { range / 3.0 } else { 0.0 };

            let sc_quant = if scale > 0.0 {
                (scale / max_d * 15.0).min(15.0) as u8
            } else {
                0
            };

            let m_quant = if min_val < 0.0 {
                ((-min_val) / max_m * 15.0).min(15.0) as u8
            } else {
                0
            };

            scales[block_idx] = (m_quant << 4) | sc_quant;

            max_d = max_d.max(scale);
            max_m = max_m.max(min_val.abs());

            for i in 0usize..4 {
                let mut byte = 0u8;
                for b in 0..4 {
                    let idx = i * 4 + b;
                    let val = block[idx];
                    let q = if scale > 0.0 {
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

        let d = f32_to_f16(max_d / 15.0);
        let dmin = f32_to_f16(max_m / 15.0);

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
