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
}
