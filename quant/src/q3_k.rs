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
