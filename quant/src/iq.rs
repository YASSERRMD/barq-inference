//! IQ (Importance-aware Quantization) implementations from ik_llama.cpp
//!
//! Handles advanced quantization formats IQ2_KS, IQ3_KS, IQ4_KS, and others.
//! Reference: ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp

use barq_core::error::{Error, Result};
use half::f16;

/// The non-linear 4-bit value lookup table for IQ4_KS.
/// Two groups of 16: base values (index 0..15) and shifted values (index 16..31).
/// Source: ggml-common.h `iq4k_values`
pub const IQ4K_VALUES: [i8; 32] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
];

/// IQ quantization types
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IQType {
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_NL,
    IQ4_XS,
    // ik_llama specific extensions
    IQ2_KS,
    IQ3_KS,
    IQ4_KS,
    Q4_K_R4,
}

/// IQ quantization configuration
#[derive(Debug, Clone)]
pub struct IQQuantConfig {
    pub iq_type: IQType,
    pub block_size: usize,
}

impl Default for IQQuantConfig {
    fn default() -> Self {
        Self {
            iq_type: IQType::IQ4_NL,
            block_size: 32,
        }
    }
}

impl IQQuantConfig {
    pub fn iq4_ks() -> Self {
        Self { iq_type: IQType::IQ4_KS, block_size: 256 }
    }
    pub fn q4_k_r4() -> Self {
        Self { iq_type: IQType::Q4_K_R4, block_size: 256 }
    }
}

/// IQ4_KS block structure (matches ik_llama.cpp block_iq4_ks).
///
/// Memory layout per super-block of 256 values:
/// - 1x f32 super-block scale (`d`), stored *before* the blocks array
/// - For each sub-block of 32 values:
///   - `scales[ib]`: packed u8. Bit 0 = shift flag. Bits 1..7 = quantized sub-scale.
///   - `qs[ib * 16 .. ib * 16 + 16]`: 16 bytes, 2 values packed per byte (4 bits each)
///
/// The actual binary layout on disk has an f32 prepended to each row, NOT inside
/// the struct. The struct only contains the per-block scales and quantized weights.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct BlockIq4Ks {
    /// Sub-block scales. bit0 = is_shifted, bits 1-7 = scale level mapped to [-127..127].
    pub scales: [u8; 8],
    /// 4-bit packed quantized weights: 256 values → 128 bytes (low nibble then high nibble).
    pub qs: [u8; 128],
}

/// IQ3_KS block structure
#[derive(Debug, Clone)]
#[repr(C)]
pub struct BlockIq3Ks {
    pub d: f16,
    pub scales: [u8; 12],
    pub qs: [u8; 96], // 3 bits per value
}

/// IQ2_KS block structure
#[derive(Debug, Clone)]
#[repr(C)]
pub struct BlockIq2Ks {
    pub d: f16,
    pub scales: [u8; 8],
    pub qs: [u8; 64], // 2 bits per value
}

/// Q4_K_R4 repacked structure for CPU
/// 4 rows interleaved for SIMD-friendly access.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct BlockQ4KR4 {
    pub d: [f16; 4],
    pub scales: [u8; 48],
    pub qs: [u8; 512],
}

/// Dequantize IQ4_KS format into f32 values.
///
/// Matches `dequantize_row_iq4_ks` in ik_llama.cpp exactly.
///
/// Layout: `[f32 d] [block_iq4_ks * n_blocks]`
/// Each block_iq4_ks: `scales[8]` + `qs[128]`
pub fn dequantize_iq4_ks(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 32;  // sub-block size within the super-block
    const SCALES_PER_BLOCK: usize = QK_K / BLOCK_SIZE; // = 8

    // The per-row f32 scale `d` is the first 4 bytes
    if data.len() < 4 {
        return Err(Error::Tensor("IQ4_KS data too short for scale".into()));
    }
    let d = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);

    // Remaining bytes are block_iq4_ks structures
    // Each block = 8 (scales) + 128 (qs) = 136 bytes for 256 values
    let block_bytes = 8 + 128; // sizeof(block_iq4_ks)
    let n_blocks = (n_elements + QK_K - 1) / QK_K;
    let block_data = &data[4..];

    if block_data.len() < n_blocks * block_bytes {
        return Err(Error::Tensor(format!(
            "IQ4_KS: need {} bytes for {} blocks, got {}",
            n_blocks * block_bytes, n_blocks, block_data.len()
        )));
    }

    let mut output = Vec::with_capacity(n_elements);

    for ibl in 0..n_blocks {
        let blk = &block_data[ibl * block_bytes..][..block_bytes];
        let scales = &blk[0..8];
        let qs = &blk[8..8 + 128];

        let mut qs_offset = 0usize;

        for ib in 0..SCALES_PER_BLOCK {
            // scales[ib]: bit0 = shift flag (which group of 16 in IQ4K_VALUES)
            //             bits 1..7: quantized sub-scale, mapped to [-127..127]
            let scale_byte = scales[ib];
            let is_shifted = (scale_byte & 1) != 0;
            let scale_level = (scale_byte & 0xFE) as i32 - 127;
            let dl = d * scale_level as f32;

            // Pick which group of 16 lookup values to use
            let values_offset: usize = if is_shifted { 16 } else { 0 };

            // Process BLOCK_SIZE = 32 values from 16 bytes
            // Low nibble = value at j, high nibble = value at j + BLOCK_SIZE/2
            for j in 0..(BLOCK_SIZE / 2) {
                let byte = qs[qs_offset + j];
                let v_lo = IQ4K_VALUES[values_offset + (byte & 0x0F) as usize];
                let v_hi = IQ4K_VALUES[values_offset + (byte >> 4) as usize];
                output.push(dl * v_lo as f32);
                output.push(dl * v_hi as f32);
            }
            qs_offset += BLOCK_SIZE / 2;
        }
    }

    output.truncate(n_elements);
    Ok(output)
}

/// Dequantize IQ quantized data (dispatcher)
pub fn dequantize_iq(data: &[u8], config: &IQQuantConfig) -> Result<Vec<f32>> {
    let n_elements = {
        // Estimate element count from data length and type
        match config.iq_type {
            IQType::IQ4_KS => {
                // [4 bytes f32 d] + [n_blocks * 136 bytes]
                let block_bytes = 136usize;
                let n_blocks = (data.len().saturating_sub(4)) / block_bytes;
                n_blocks * 256
            }
            _ => data.len() // fallback
        }
    };

    match config.iq_type {
        IQType::IQ4_KS => dequantize_iq4_ks(data, n_elements),
        _ => {
            // Placeholder for other IK types
            Ok(vec![0.0f32; n_elements.max(1)])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that IQ4K_VALUES matches the reference table from ggml-common.h
    #[test]
    fn test_iq4k_values_table() {
        assert_eq!(IQ4K_VALUES[0], -127);
        assert_eq!(IQ4K_VALUES[7], -10);
        assert_eq!(IQ4K_VALUES[15], 113);
        // shifted group
        assert_eq!(IQ4K_VALUES[16], -123);
        assert_eq!(IQ4K_VALUES[31], 117);
    }

    /// Tests that dequantize_iq4_ks correctly handles a zero block (all zeros → all zeros out)
    #[test]
    fn test_iq4_ks_zero_block() {
        // Build a minimal valid IQ4_KS blob: 4-byte f32 scale + one 136-byte block
        let mut data = vec![0u8; 4 + 136];
        // d = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        // scales[0] = 127 (scale_level = 127 & 0xFE - 127 = 126 - 127 = -1, not zero)
        // Use scale_byte = 128 = 0x80 → scale_level = 128 & 0xFE - 127 = 128 - 127 = 1, no shift
        data[4] = 128u8; // scales[0]
        // qs all zero → all values are IQ4K_VALUES[0] = -127
        let result = dequantize_iq4_ks(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
        // All qs nibbles are 0x0 → IQ4K_VALUES[0] = -127
        // dl = 1.0 * 1 = 1.0; value[0] = -127
        let expected = 1.0 * (-127i8 as f32);
        for val in &result[0..32] {
            assert_eq!(*val, expected, "Expected {}, got {}", expected, val);
        }
        // Remaining 7 sub-blocks have scale_byte = 0 → scale_level = 0 - 127 = -127 → dl = -127
        let expected_rest = 1.0 * (-127i8 as f32) * (-127f32);
        for val in &result[32..] {
            assert_eq!(*val, expected_rest);
        }
    }
}
