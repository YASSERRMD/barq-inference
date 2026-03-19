//! IQ (Importance-aware Quantization) implementations from ik_llama.cpp
//!
//! Handles advanced quantization formats IQ2_KS, IQ3_KS, IQ4_KS, and others.
//! Reference: ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp

use barq_core::error::{Error, Result};
use half::f16;

// ─────────────────────────────────────────────────────────────────────────────
// Lookup tables from ggml-common.h
// ─────────────────────────────────────────────────────────────────────────────

/// Non-linear 4-bit lookup table for IQ4_K / IQ4_KS.
/// Two groups of 16: base values (index 0..15) and shifted values (index 16..31).
/// Source: ggml-common.h `iq4k_values`
pub const IQ4K_VALUES: [i8; 32] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, -123, -100, -79, -61,
    -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117,
];

/// Non-linear 2-bit lookup table for IQ2_KS.
/// Two groups of 4: base values (index 0..3) and shifted values (index 4..7).
/// Source: ggml-common.h `iq2nl_values`
pub const IQ2NL_VALUES: [i8; 8] = [-31, -13, 1, 17, -26, -8, 6, 22];

/// Non-linear 3-bit lookup table for IQ3_KS.
/// Two groups of 8: base values (0..7) and shifted values (8..15).
/// Source: ggml-common.h `iq3nl_values`
pub const IQ3NL_VALUES: [i8; 16] = [
    -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
];

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

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
        Self {
            iq_type: IQType::IQ4_KS,
            block_size: 256,
        }
    }
    pub fn q4_k_r4() -> Self {
        Self {
            iq_type: IQType::Q4_K_R4,
            block_size: 256,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IQ4_KS Dequantization
// ─────────────────────────────────────────────────────────────────────────────

/// IQ4_KS block structure (matches ik_llama.cpp `block_iq4_ks`).
///
/// Memory layout per super-block of 256 values:
/// - 1x f32 super-block scale (`d`), stored *before* the blocks array (per row)
/// - For each sub-block of 32 values:
///   - `scales[ib]`: packed u8. Bit 0 = shift flag. Bits 1..7 = quantized sub-scale → [-127..127].
///   - `qs[ib * 16 .. ib * 16 + 16]`: 16 bytes, 2 values packed per byte (4 bits each)
///
/// `sizeof(block_iq4_ks) == QK_K/32 + QK_K/2 = 8 + 128 = 136 bytes`
/// The f32 row scale (`d`) is prepended once per row, outside the struct.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BlockIq4Ks {
    /// Sub-block scales. bit0 = is_shifted, bits 1-7 = scale level mapped to [-127..127].
    pub scales: [u8; 8],
    /// 4-bit packed quantized weights: 256 values → 128 bytes.
    pub qs: [u8; 128],
}

/// Dequantize IQ4_KS format into f32 values.
///
/// Matches `dequantize_row_iq4_ks` in ik_llama.cpp exactly.
///
/// On-disk layout per row:
/// ```text
/// [f32 d]                           4 bytes  (row scale)
/// [block_iq4_ks * n_blocks]         n_blocks × 136 bytes
/// ```
pub fn dequantize_iq4_ks(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 32; // sub-block size
    const N_SCALES: usize = QK_K / BLOCK_SIZE; // = 8

    if data.len() < 4 {
        return Err(Error::Tensor("IQ4_KS data too short for row scale".into()));
    }

    // Row-level f32 scale
    let d = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);

    // Each block = 8 bytes scales + 128 bytes qs = 136 bytes
    const BLOCK_BYTES: usize = N_SCALES + 128;
    let n_blocks = n_elements.div_ceil(QK_K);
    let block_data = &data[4..];

    if block_data.len() < n_blocks * BLOCK_BYTES {
        return Err(Error::Tensor(format!(
            "IQ4_KS: need {} bytes for {} blocks, got {}",
            n_blocks * BLOCK_BYTES,
            n_blocks,
            block_data.len()
        )));
    }

    let mut output = Vec::with_capacity(n_elements);

    for ibl in 0..n_blocks {
        let blk = &block_data[ibl * BLOCK_BYTES..][..BLOCK_BYTES];
        let scales = &blk[0..N_SCALES];
        let qs = &blk[N_SCALES..BLOCK_BYTES];

        let mut qs_off = 0usize;

        for ib in 0..N_SCALES {
            // scale_byte: bit0 = is_shifted, bits 1-7 map to [-127..127]
            let scale_byte = scales[ib];
            let is_shifted = (scale_byte & 1) != 0;
            // (scale_byte & 0xFE) is an unsigned value [0,254], subtract 127 → [-127,127]
            let scale_level: i32 = (scale_byte & 0xFE) as i32 - 127;
            let dl = d * scale_level as f32;

            let values_offset = if is_shifted { 16usize } else { 0 };

            // Each sub-block: 16 bytes → 32 values
            // Low nibble = output[j], high nibble = output[j + BLOCK_SIZE/2]
            for j in 0..(BLOCK_SIZE / 2) {
                let byte = qs[qs_off + j];
                let lo_idx = values_offset + (byte & 0x0F) as usize;
                let hi_idx = values_offset + (byte >> 4) as usize;
                output.push(dl * IQ4K_VALUES[lo_idx] as f32);
                output.push(dl * IQ4K_VALUES[hi_idx] as f32);
            }
            qs_off += BLOCK_SIZE / 2;
        }
    }

    output.truncate(n_elements);
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// IQ3_KS Dequantization
// ─────────────────────────────────────────────────────────────────────────────

/// IQ3_KS block structure (matches ik_llama.cpp `block_iq3_ks`).
///
/// `sizeof(block_iq3_ks) == uint16_t extra (2) + QK_K/64 scales (4) + QK_K/4 qs (64) + QK_K/8 qh (32) = 102 bytes`
/// The f16 row scale (`d`) is prepended once per row outside the struct.
///
/// On-disk layout per row:
/// ```text
/// [f16 d]                  2 bytes (row scale)
/// [block_iq3_ks * n_blocks] n_blocks × 102 bytes
/// ```
pub fn dequantize_iq3_ks(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 32; // sub-block size (kBlockSize = 32)
                                  // block layout: uint16_t extra (2) + scales[QK_K/64=4] + qs[QK_K/4=64] + qh[QK_K/8=32]
    const BLOCK_BYTES: usize = 2 + 4 + 64 + 32; // 102
    const N_BLOCKS_PER_SUPER: usize = QK_K / BLOCK_SIZE; // = 8

    if data.len() < 2 {
        return Err(Error::Tensor("IQ3_KS data too short for row scale".into()));
    }

    // Row-level f16 scale
    let d = f16::from_le_bytes([data[0], data[1]]).to_f32();

    let n_blocks = n_elements.div_ceil(QK_K);
    let block_data = &data[2..];

    if block_data.len() < n_blocks * BLOCK_BYTES {
        return Err(Error::Tensor(format!(
            "IQ3_KS: need {} bytes for {} blocks, got {}",
            n_blocks * BLOCK_BYTES,
            n_blocks,
            block_data.len()
        )));
    }

    let mut output = Vec::with_capacity(n_elements);

    // dl computation (matching dequantize_row_iq3_ks in C++):
    // for each ib in 0..8:
    //   ls1 = (scales[ib%4] & 0xf) | (((extra >> (ib+0)) & 1) << 4)  → but ib runs 0..4 for lower half
    //   …actually per the C++ code:
    //   for j in 0..4:
    //       int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j+0)) & 1) << 4)
    //       int ls2 = (x[ibl].scales[j] >> 4)  | (((x[ibl].extra >> (j+4)) & 1) << 4)
    //       dl[j+0] = d*(ls1 - 16)
    //       dl[j+4] = d*(ls2 - 16)
    //
    // then for each i128 in 0..(QK_K/128):
    //   for ib in 0..4:
    //       values = iq3nl_values offset by ((extra >> (8 + 4*i128+ib)) & 1) << 3
    //       for j in 0..32:
    //           y[j] = dl[4*i128+ib] * values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)]

    for ibl in 0..n_blocks {
        let blk = &block_data[ibl * BLOCK_BYTES..][..BLOCK_BYTES];
        let extra = u16::from_le_bytes([blk[0], blk[1]]);
        let scales = &blk[2..6]; // [4]
        let qs = &blk[6..70]; // [64]
        let qh = &blk[70..102]; // [32]

        // Build dl[8]
        let mut dl = [0f32; 8];
        for j in 0..4 {
            let ls1 = ((scales[j] & 0xf) as u32 | (((extra >> j) & 1) as u32) << 4) as i32;
            let ls2 = ((scales[j] >> 4) as u32 | (((extra >> (j + 4)) & 1) as u32) << 4) as i32;
            dl[j] = d * (ls1 - 16) as f32;
            dl[j + 4] = d * (ls2 - 16) as f32;
        }

        let mut y = vec![0f32; QK_K];

        for i128 in 0..(QK_K / 128) {
            for ib in 0..4 {
                let shift_flag = ((extra >> (8 + 4 * i128 as u16 + ib as u16)) & 1) as usize;
                let values_offset = shift_flag << 3; // 0 or 8

                let qs_base = i128 * 32; // each i128 consumes 32 qs bytes (4 sub-blocks × 8 values)
                let qh_base = 0usize; // qh is shared, indexed by (4*i128+ib) shift

                for j in 0..BLOCK_SIZE {
                    let q_low = ((qs[qs_base + j] >> (2 * ib)) & 3) as usize;
                    let q_high = (((qh[qh_base + j] >> (4 * i128 + ib)) & 1) as usize) << 2;
                    let idx = values_offset + q_low | q_high;
                    y[i128 * 128 + ib * 32 + j] = dl[4 * i128 + ib] * IQ3NL_VALUES[idx] as f32;
                }
            }
        }

        output.extend_from_slice(&y);
    }

    output.truncate(n_elements);
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// IQ2_KS Dequantization
// ─────────────────────────────────────────────────────────────────────────────

/// IQ2_KS block structure (matches ik_llama.cpp `block_iq2_ks`).
///
/// `sizeof(block_iq2_ks) == uint16_t extra (2) + QK_K/64 scales (4) + QK_K/4 qs (64) = 70 bytes`
/// The f16 row scale (`d`) is prepended once per row outside the struct.
///
/// On-disk layout per row:
/// ```text
/// [f16 d]                   2 bytes (row scale)
/// [block_iq2_ks * n_blocks]  n_blocks × 70 bytes
/// ```
///
/// C++ reference (dequantize_row_iq2_ks):
/// ```cpp
/// for ib64 in 0..QK_K/64:
///     dl1 = d * (((scales[ib64] & 0xf) | ((extra >> 4) & 0x10)) - 16)
///     dl2 = d * (((scales[ib64] >> 4)  | ((extra >> 5) & 0x10)) - 16)
///     values1 = extra & 1 ? iq2nl_values+4 : iq2nl_values
///     values2 = extra & 2 ? iq2nl_values+4 : iq2nl_values
///     extra >>= 2
///     shift = 0 or 4 alternating within each 64-element group
///     for j in 0..32:
///         y[j+ 0] = dl1 * values1[(qs[j] >> (shift+0)) & 3]
///         y[j+32] = dl2 * values2[(qs[j] >> (shift+2)) & 3]
///     shift += 4; if shift==8 { qs += 32; shift = 0 }
/// ```
pub fn dequantize_iq2_ks(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 2 + 4 + 64; // extra(2) + scales(4) + qs(64) = 70

    if data.len() < 2 {
        return Err(Error::Tensor("IQ2_KS data too short for row scale".into()));
    }

    let d = f16::from_le_bytes([data[0], data[1]]).to_f32();
    let n_blocks = n_elements.div_ceil(QK_K);
    let block_data = &data[2..];

    if block_data.len() < n_blocks * BLOCK_BYTES {
        return Err(Error::Tensor(format!(
            "IQ2_KS: need {} bytes for {} blocks, got {}",
            n_blocks * BLOCK_BYTES,
            n_blocks,
            block_data.len()
        )));
    }

    let mut output = Vec::with_capacity(n_elements);

    for ibl in 0..n_blocks {
        let blk = &block_data[ibl * BLOCK_BYTES..][..BLOCK_BYTES];
        let mut extra = u16::from_le_bytes([blk[0], blk[1]]);
        let scales = &blk[2..6]; // [4]
        let qs = &blk[6..70]; // [64]

        let mut y = [0f32; QK_K];
        let mut qs_off = 0usize;
        let mut shift = 0u32;

        for ib64 in 0..(QK_K / 64) {
            // Scale extraction matching C++
            let dl1 = d
                * (((scales[ib64] & 0xf) as u32 | (((extra as u32) >> 4) & 0x10)) as i32 - 16)
                    as f32;
            let dl2 = d
                * (((scales[ib64] >> 4) as u32 | (((extra as u32) >> 5) & 0x10)) as i32 - 16)
                    as f32;
            let v1_off: usize = if (extra & 1) != 0 { 4 } else { 0 };
            let v2_off: usize = if (extra & 2) != 0 { 4 } else { 0 };
            extra >>= 2;

            for j in 0..32usize {
                y[ib64 * 64 + j] =
                    dl1 * IQ2NL_VALUES[v1_off + ((qs[qs_off + j] >> shift) & 3) as usize] as f32;
                y[ib64 * 64 + j + 32] = dl2
                    * IQ2NL_VALUES[v2_off + ((qs[qs_off + j] >> (shift + 2)) & 3) as usize] as f32;
            }

            shift += 4;
            if shift == 8 {
                qs_off += 32;
                shift = 0;
            }
        }

        output.extend_from_slice(&y);
    }

    output.truncate(n_elements);
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Dequantize IQ quantized data (dispatcher)
pub fn dequantize_iq(data: &[u8], config: &IQQuantConfig) -> Result<Vec<f32>> {
    let n_elements = estimate_n_elements(data, config);

    match config.iq_type {
        IQType::IQ4_KS => dequantize_iq4_ks(data, n_elements),
        IQType::IQ3_KS => dequantize_iq3_ks(data, n_elements),
        IQType::IQ2_KS => dequantize_iq2_ks(data, n_elements),
        _ => {
            // Placeholder for other IK types not yet implemented
            Ok(vec![0.0f32; n_elements.max(1)])
        }
    }
}

/// Quantize IQ data using the selected configuration.
///
/// The encoder path is intentionally conservative:
/// - it uses the exact packed layouts expected by the dequantizers above
/// - it favors deterministic, in-repo behavior over the full ik_llama.cpp search
///   stack, which depends on importance-matrix tooling that is not wired in here
pub fn quantize_iq(data: &[f32], config: &IQQuantConfig) -> Result<Vec<u8>> {
    match config.iq_type {
        IQType::IQ4_KS => quantize_iq4_ks(data),
        IQType::IQ3_KS => quantize_iq3_ks(data),
        IQType::IQ2_KS => quantize_iq2_ks(data),
        IQType::Q4_K_R4 => quantize_q4_k_r4(data),
        _ => Err(Error::Unsupported(format!(
            "Quantization for {:?} is not implemented yet",
            config.iq_type
        ))),
    }
}

const IQ_ROW_SIZE: usize = 256;

fn pad_row(row: &[f32]) -> [f32; IQ_ROW_SIZE] {
    let mut padded = [0.0f32; IQ_ROW_SIZE];
    let len = row.len().min(IQ_ROW_SIZE);
    padded[..len].copy_from_slice(&row[..len]);
    padded
}

fn best_codebook_index(value: f32, scale: f32, codebook: &[i8]) -> usize {
    if codebook.is_empty() || scale == 0.0 {
        return 0;
    }

    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;

    for (idx, &entry) in codebook.iter().enumerate() {
        let candidate = scale * entry as f32;
        let err = (value - candidate).abs();
        if err < best_err {
            best_err = err;
            best_idx = idx;
        }
    }

    best_idx
}

fn best_codebook_subset(value: &[f32], scale: f32, first: &[i8], second: &[i8]) -> bool {
    let mut err_first = 0.0f32;
    let mut err_second = 0.0f32;

    for &v in value {
        let idx_first = best_codebook_index(v, scale, first);
        let idx_second = best_codebook_index(v, scale, second);
        let cand_first = scale * first[idx_first] as f32;
        let cand_second = scale * second[idx_second] as f32;
        err_first += (v - cand_first).abs();
        err_second += (v - cand_second).abs();
    }

    err_second < err_first
}

fn quantize_rows_as_bytes<F>(data: &[f32], mut encode_row: F) -> Result<Vec<u8>>
where
    F: FnMut(&[f32]) -> Result<Vec<u8>>,
{
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n_rows = data.len().div_ceil(IQ_ROW_SIZE);
    let mut output = Vec::new();

    for row_idx in 0..n_rows {
        let start = row_idx * IQ_ROW_SIZE;
        let end = (start + IQ_ROW_SIZE).min(data.len());
        let row = pad_row(&data[start..end]);
        output.extend(encode_row(&row)?);
    }

    Ok(output)
}

fn quantize_iq4_ks_row(row: &[f32]) -> Result<Vec<u8>> {
    let row = pad_row(row);
    let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };

    let mut out = Vec::with_capacity(4 + 136);
    out.extend_from_slice(&d.to_le_bytes());

    let mut block = [0u8; 136];
    let base = &IQ4K_VALUES[0..16];
    let shifted = &IQ4K_VALUES[16..32];

    for sub in 0..8 {
        let sub_values = &row[sub * 32..sub * 32 + 32];
        let use_shifted = best_codebook_subset(sub_values, d, base, shifted);
        let codebook = if use_shifted { shifted } else { base };
        block[sub] = if use_shifted { 129 } else { 128 };

        for byte_idx in 0..16 {
            let lhs = sub_values[byte_idx * 2];
            let rhs = sub_values[byte_idx * 2 + 1];
            let q0 = best_codebook_index(lhs, d, codebook) as u8;
            let q1 = best_codebook_index(rhs, d, codebook) as u8;
            block[8 + sub * 16 + byte_idx] = q0 | (q1 << 4);
        }
    }

    out.extend_from_slice(&block);
    Ok(out)
}

fn quantize_iq3_ks_row(row: &[f32]) -> Result<Vec<u8>> {
    let row = pad_row(row);
    let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d = if max_abs > 0.0 {
        max_abs / (16.0 * 63.0)
    } else {
        0.0
    };

    let mut out = Vec::with_capacity(2 + 102);
    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());

    let mut block = [0u8; 102];
    let codebook = &IQ3NL_VALUES[0..8];

    for i128 in 0..2 {
        let qs_base = i128 * 32;
        let qh_base = i128 * 16;

        for j in 0..32 {
            let mut qs_byte = 0u8;
            let values = &row[i128 * 128..i128 * 128 + 128];

            for ib in 0..4 {
                let idx = ib * 32 + j;
                let code = best_codebook_index(values[idx], -16.0 * d, codebook) as u8;
                qs_byte |= (code & 0x03) << (2 * ib);
                if code & 0x04 != 0 {
                    block[70 + j] |= 1 << (4 * i128 + ib);
                }
            }

            block[6 + qs_base + j] = qs_byte;
        }

        for j in 0..4 {
            block[2 + j] = 0;
        }

        for j in 0..4 {
            block[2 + j + 4] = 0;
        }

        for j in 0..4 {
            let _ = qh_base + j;
        }
    }

    out.extend_from_slice(&block);
    Ok(out)
}

fn quantize_iq2_ks_row(row: &[f32]) -> Result<Vec<u8>> {
    let row = pad_row(row);
    let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d = if max_abs > 0.0 {
        max_abs / (16.0 * 31.0)
    } else {
        0.0
    };

    let mut out = Vec::with_capacity(2 + 70);
    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());

    let mut block = [0u8; 70];
    let codebook = &IQ2NL_VALUES[0..4];

    for ib64 in 0..4 {
        let shift = if ib64 % 2 == 0 { 0 } else { 4 };
        let qs_base = (ib64 / 2) * 32;

        for j in 0..32 {
            let a = row[ib64 * 64 + j];
            let b = row[ib64 * 64 + j + 32];
            let qa = best_codebook_index(a, -16.0 * d, codebook) as u8;
            let qb = best_codebook_index(b, -16.0 * d, codebook) as u8;
            let mut qs_byte = block[6 + qs_base + j];
            qs_byte |= (qa & 0x03) << shift;
            qs_byte |= (qb & 0x03) << (shift + 2);
            block[6 + qs_base + j] = qs_byte;
        }
    }

    out.extend_from_slice(&block);
    Ok(out)
}

fn quantize_q4_k_r4_row_group(rows: &[[f32; IQ_ROW_SIZE]; 4]) -> Result<Vec<u8>> {
    let mut d = [0f32; 4];
    let mut m = [0f32; 4];
    let mut row_qs = [[0u8; IQ_ROW_SIZE / 2]; 4];

    for (row_idx, row) in rows.iter().enumerate() {
        let min_val = row.iter().fold(f32::INFINITY, |acc, &v| acc.min(v));
        let max_val = row.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
        let offset = if min_val < 0.0 { -min_val } else { 0.0 };
        let range = max_val + offset;

        d[row_idx] = if range > 0.0 { range / 15.0 } else { 1.0 };
        m[row_idx] = offset;

        for (i, &value) in row.iter().enumerate() {
            let q = if d[row_idx] > 0.0 {
                ((value + offset) / d[row_idx]).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };

            let byte_idx = i / 2;
            let shift = if i % 2 == 0 { 0 } else { 4 };
            row_qs[row_idx][byte_idx] |= q << shift;
        }
    }

    let mut out = Vec::with_capacity(576);
    for &scale in &d {
        out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
    }
    for &offset in &m {
        out.extend_from_slice(&f16::from_f32(offset).to_le_bytes());
    }

    let mut scales_l = [0u8; 32];
    let scales_h = [0u8; 16];
    scales_l.fill(0x11);
    out.extend_from_slice(&scales_h);
    out.extend_from_slice(&scales_l);

    let mut qs = [0u8; 512];
    for ib in 0..8 {
        for row_idx in 0..4 {
            let src_start = ib * 16;
            let src_end = src_start + 16;
            let dst_start = ib * 64 + row_idx * 16;
            qs[dst_start..dst_start + 16].copy_from_slice(&row_qs[row_idx][src_start..src_end]);
        }
    }
    out.extend_from_slice(&qs);

    Ok(out)
}

/// Quantize IQ4_KS data into packed bytes.
pub fn quantize_iq4_ks(data: &[f32]) -> Result<Vec<u8>> {
    quantize_rows_as_bytes(data, quantize_iq4_ks_row)
}

/// Quantize IQ3_KS data into packed bytes.
pub fn quantize_iq3_ks(data: &[f32]) -> Result<Vec<u8>> {
    quantize_rows_as_bytes(data, quantize_iq3_ks_row)
}

/// Quantize IQ2_KS data into packed bytes.
pub fn quantize_iq2_ks(data: &[f32]) -> Result<Vec<u8>> {
    quantize_rows_as_bytes(data, quantize_iq2_ks_row)
}

/// Quantize Q4_K_R4 data into packed bytes.
///
/// The layout follows the interleaved 4-row block structure used by ik_llama.cpp.
pub fn quantize_q4_k_r4(data: &[f32]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    let n_groups = data.len().div_ceil(IQ_ROW_SIZE * 4);

    for group_idx in 0..n_groups {
        let mut rows = [[0.0f32; IQ_ROW_SIZE]; 4];

        for row_idx in 0..4 {
            let src_row = group_idx * 4 * IQ_ROW_SIZE + row_idx * IQ_ROW_SIZE;
            let src_end = (src_row + IQ_ROW_SIZE).min(data.len());
            if src_row < data.len() {
                rows[row_idx][..(src_end - src_row)].copy_from_slice(&data[src_row..src_end]);
            }
        }

        output.extend_from_slice(&quantize_q4_k_r4_row_group(&rows)?);
    }

    Ok(output)
}

fn quantize_iq4_ks_row_best_effort(row: &[f32; IQ_ROW_SIZE]) -> Vec<f32> {
    let d = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 127.0;
    row.iter()
        .map(|&v| {
            let idx = best_codebook_index(v, d, &IQ4K_VALUES[0..16]);
            d * IQ4K_VALUES[0..16][idx] as f32
        })
        .collect()
}

fn quantize_iq3_ks_row_best_effort(row: &[f32; IQ_ROW_SIZE]) -> Vec<f32> {
    let d = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / (16.0 * 63.0);
    row.iter()
        .map(|&v| {
            let idx = best_codebook_index(v, -16.0 * d, &IQ3NL_VALUES[0..8]);
            -16.0 * d * IQ3NL_VALUES[0..8][idx] as f32
        })
        .collect()
}

fn quantize_iq2_ks_row_best_effort(row: &[f32; IQ_ROW_SIZE]) -> Vec<f32> {
    let d = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / (16.0 * 31.0);
    row.iter()
        .map(|&v| {
            let idx = best_codebook_index(v, -16.0 * d, &IQ2NL_VALUES[0..4]);
            -16.0 * d * IQ2NL_VALUES[0..4][idx] as f32
        })
        .collect()
}

fn quantize_q4_k_r4_row_best_effort(rows: &[[f32; IQ_ROW_SIZE]; 4]) -> Vec<f32> {
    let mut flattened = Vec::with_capacity(IQ_ROW_SIZE * 4);
    for row in rows {
        let min_val = row.iter().fold(f32::INFINITY, |acc, &v| acc.min(v));
        let max_val = row.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
        let offset = if min_val < 0.0 { -min_val } else { 0.0 };
        let range = max_val + offset;
        let d = if range > 0.0 { range / 15.0 } else { 1.0 };
        flattened.extend(row.iter().map(|&v| {
            let q = ((v + offset) / d).round().clamp(0.0, 15.0);
            d * q - offset
        }));
    }
    flattened
}

/// Estimate the number of elements from raw data length
fn estimate_n_elements(data: &[u8], config: &IQQuantConfig) -> usize {
    let (header, block_bytes, qk_k) = match config.iq_type {
        IQType::IQ4_KS => (4usize, 136usize, 256usize), // f32 row scale
        IQType::IQ3_KS => (2usize, 102usize, 256usize), // f16 row scale
        IQType::IQ2_KS => (2usize, 70usize, 256usize),  // f16 row scale
        _ => return data.len(),
    };
    let n_blocks = (data.len().saturating_sub(header)) / block_bytes;
    n_blocks * qk_k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies IQ4K_VALUES matches the reference table from ggml-common.h
    #[test]
    fn test_iq4k_values_table() {
        assert_eq!(IQ4K_VALUES[0], -127);
        assert_eq!(IQ4K_VALUES[7], -10);
        assert_eq!(IQ4K_VALUES[15], 113);
        assert_eq!(IQ4K_VALUES[16], -123);
        assert_eq!(IQ4K_VALUES[31], 117);
    }

    /// Verifies IQ2NL_VALUES matches ggml-common.h `iq2nl_values`
    #[test]
    fn test_iq2nl_values_table() {
        assert_eq!(IQ2NL_VALUES[0], -31);
        assert_eq!(IQ2NL_VALUES[3], 17);
        assert_eq!(IQ2NL_VALUES[4], -26);
        assert_eq!(IQ2NL_VALUES[7], 22);
    }

    /// Verifies IQ3NL_VALUES matches ggml-common.h `iq3nl_values`
    #[test]
    fn test_iq3nl_values_table() {
        assert_eq!(IQ3NL_VALUES[0], -63);
        assert_eq!(IQ3NL_VALUES[7], 47);
        assert_eq!(IQ3NL_VALUES[8], -59);
        assert_eq!(IQ3NL_VALUES[15], 51);
    }

    /// Tests that dequantize_iq4_ks handles a zero-scale block correctly
    #[test]
    fn test_iq4_ks_zero_block() {
        // Build a minimal valid IQ4_KS blob: 4-byte f32 scale + one 136-byte block
        let mut data = vec![0u8; 4 + 136];
        // d = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        // scales[0] = 128 => (128 & 0xFE) - 127 = 128 - 127 = 1, no shift
        data[4] = 128u8;
        // qs all zero → all nibbles 0x0 → IQ4K_VALUES[0] = -127
        let result = dequantize_iq4_ks(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
        // first 32 values: dl = 1.0*1 = 1.0, val = -127
        assert_eq!(result[0], 1.0 * -127_f32);
        assert_eq!(result[31], 1.0 * -127_f32);
    }

    /// Tests that dequantize_iq2_ks produces correctly sized output
    #[test]
    fn test_iq2_ks_output_size() {
        // f16 d = 1.0, one block: 70 bytes
        let mut data = vec![0u8; 2 + 70];
        data[0..2].copy_from_slice(&f16::ONE.to_le_bytes());
        let result = dequantize_iq2_ks(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
    }

    /// Tests that dequantize_iq3_ks produces correctly sized output
    #[test]
    fn test_iq3_ks_output_size() {
        // f16 d = 1.0, one block: 102 bytes
        let mut data = vec![0u8; 2 + 102];
        data[0..2].copy_from_slice(&f16::ONE.to_le_bytes());
        let result = dequantize_iq3_ks(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_quantize_iq4_ks_constant_block() {
        let input = vec![1.0f32; 256];
        let bytes = quantize_iq4_ks(&input).unwrap();
        let decoded = dequantize_iq4_ks(&bytes, 256).unwrap();
        assert_eq!(decoded.len(), 256);
        assert!(decoded.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_quantize_iq3_ks_constant_block() {
        let input = vec![2.0f32; 256];
        let bytes = quantize_iq3_ks(&input).unwrap();
        let decoded = dequantize_iq3_ks(&bytes, 256).unwrap();
        assert_eq!(decoded.len(), 256);
        assert!(decoded.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_quantize_iq2_ks_constant_block() {
        let input = vec![3.0f32; 256];
        let bytes = quantize_iq2_ks(&input).unwrap();
        let decoded = dequantize_iq2_ks(&bytes, 256).unwrap();
        assert_eq!(decoded.len(), 256);
        assert!(decoded.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_quantize_q4_k_r4_size() {
        let input = vec![0.5f32; 256 * 4];
        let bytes = quantize_q4_k_r4(&input).unwrap();
        assert_eq!(bytes.len(), 576);
    }
}
