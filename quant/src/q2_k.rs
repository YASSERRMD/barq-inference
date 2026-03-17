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
