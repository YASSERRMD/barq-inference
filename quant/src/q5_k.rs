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
}
