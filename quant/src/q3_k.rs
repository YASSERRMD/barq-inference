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
}
