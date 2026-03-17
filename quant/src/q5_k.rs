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
