//! SIMD-optimized dequantization for Q4_0 and Q4_K formats
//!
//! This module provides accelerated dequantization using SIMD intrinsics

use barq_core::error::Result;

/// Dequantize Q4_0 block using SIMD
///
/// Processes multiple blocks in parallel using AVX2/NEON when available
pub fn dequantize_q4_0_simd(
    quants: &[u8],
    scales: &[f32],
    block_size: usize,
    output: &mut [f32],
) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            if is_x86_feature_detected!("avx2") {
                return dequantize_q4_0_avx2(quants, scales, block_size, output);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            if is_aarch64_feature_detected!("neon") {
                return dequantize_q4_0_neon(quants, scales, block_size, output);
            }
        }
    }

    // Scalar fallback
    dequantize_q4_0_scalar(quants, scales, block_size, output)
}

fn dequantize_q4_0_scalar(
    quants: &[u8],
    scales: &[f32],
    block_size: usize,
    output: &mut [f32],
) -> Result<()> {
    let n_blocks = scales.len();
    let mut q_offset = 0;

    for block_idx in 0..n_blocks {
        let scale = scales[block_idx];
        let start = block_idx * block_size;
        let end = (start + block_size).min(output.len());

        for i in start..end {
            let rel_idx = i - start;
            let byte_idx = rel_idx / 2;
            let shift = if rel_idx % 2 == 0 { 0 } else { 4 };

            if q_offset + byte_idx < quants.len() {
                let q = ((quants[q_offset + byte_idx] >> shift) & 0x0F) as i8;
                let q = if q >= 8 { q - 16 } else { q };
                output[i] = q as f32 * scale;
            }
        }

        q_offset += (block_size + 1) / 2;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_q4_0_avx2(
    quants: &[u8],
    scales: &[f32],
    block_size: usize,
    output: &mut [f32],
) -> Result<()> {
    use std::arch::x86_64::*;

    let n_blocks = scales.len();
    let mut q_offset = 0;

    for block_idx in 0..n_blocks {
        let scale = scales[block_idx];
        let start = block_idx * block_size;
        let end = (start + block_size).min(output.len());

        // Broadcast scale to all lanes
        let scale_vec = _mm256_set1_ps(scale);

        let mut i = start;
        // Process 8 values at a time
        while i + 8 <= end {
            let mut deq_vals = [0.0f32; 8];

            for j in 0..8 {
                let rel_idx = (i + j) - start;
                let byte_idx = rel_idx / 2;
                let shift = if rel_idx % 2 == 0 { 0 } else { 4 };

                if q_offset + byte_idx < quants.len() {
                    let q = ((quants[q_offset + byte_idx] >> shift) & 0x0F) as i8;
                    let q = if q >= 8 { q - 16 } else { q };
                    deq_vals[j] = q as f32;
                }
            }

            // Load and multiply by scale
            let vals = _mm256_loadu_ps(deq_vals.as_ptr());
            let result = _mm256_mul_ps(vals, scale_vec);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);

            i += 8;
        }

        // Process remaining values
        while i < end {
            let rel_idx = i - start;
            let byte_idx = rel_idx / 2;
            let shift = if rel_idx % 2 == 0 { 0 } else { 4 };

            if q_offset + byte_idx < quants.len() {
                let q = ((quants[q_offset + byte_idx] >> shift) & 0x0F) as i8;
                let q = if q >= 8 { q - 16 } else { q };
                output[i] = q as f32 * scale;
            }
            i += 1;
        }

        q_offset += (block_size + 1) / 2;
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequantize_q4_0_neon(
    quants: &[u8],
    scales: &[f32],
    block_size: usize,
    output: &mut [f32],
) -> Result<()> {
    use std::arch::aarch64::*;

    let n_blocks = scales.len();
    let mut q_offset = 0;

    for block_idx in 0..n_blocks {
        let scale = scales[block_idx];
        let start = block_idx * block_size;
        let end = (start + block_size).min(output.len());

        // Broadcast scale to all lanes
        let scale_vec = vdupq_n_f32(scale);

        let mut i = start;
        // Process 4 values at a time
        while i + 4 <= end {
            let mut deq_vals = [0.0f32; 4];

            for j in 0..4 {
                let rel_idx = (i + j) - start;
                let byte_idx = rel_idx / 2;
                let shift = if rel_idx % 2 == 0 { 0 } else { 4 };

                if q_offset + byte_idx < quants.len() {
                    let q = ((quants[q_offset + byte_idx] >> shift) & 0x0F) as i8;
                    let q = if q >= 8 { q - 16 } else { q };
                    deq_vals[j] = q as f32;
                }
            }

            // Load and multiply by scale
            let vals = vld1q_f32(deq_vals.as_ptr());
            let result = vmulq_f32(vals, scale_vec);
            vst1q_f32(output.as_mut_ptr().add(i), result);

            i += 4;
        }

        // Process remaining values
        while i < end {
            let rel_idx = i - start;
            let byte_idx = rel_idx / 2;
            let shift = if rel_idx % 2 == 0 { 0 } else { 4 };

            if q_offset + byte_idx < quants.len() {
                let q = ((quants[q_offset + byte_idx] >> shift) & 0x0F) as i8;
                let q = if q >= 8 { q - 16 } else { q };
                output[i] = q as f32 * scale;
            }
            i += 1;
        }

        q_offset += (block_size + 1) / 2;
    }

    Ok(())
}

/// Matrix multiplication with Q4_0 quantization using SIMD
///
/// Computes C = A @ B where B is Q4_0 quantized
pub fn matmul_q4_0_simd(
    a: &[f32],
    b_quants: &[u8],
    b_scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    output: &mut [f32],
) -> Result<()> {
    // For each row of A and each column of B
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;

            for l in 0..k {
                let a_val = a[i * k + l];

                // Get B value from quantized format
                let b_block_idx = (l * n + j) / 32;
                let b_pos_in_block = (l * n + j) % 32;
                let b_byte_idx = b_pos_in_block / 2;
                let b_shift = if b_pos_in_block % 2 == 0 { 0 } else { 4 };

                if b_block_idx < b_scales.len() {
                    let b_scale = b_scales[b_block_idx];
                    let q_offset = b_block_idx * 16; // 32 quants packed into 16 bytes

                    if q_offset + b_byte_idx < b_quants.len() {
                        let q = ((b_quants[q_offset + b_byte_idx] >> b_shift) & 0x0F) as i8;
                        let q = if q >= 8 { q - 16 } else { q };
                        let b_val = q as f32 * b_scale;
                        sum += a_val * b_val;
                    }
                }
            }

            output[i * n + j] = sum;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q4_0_simd_correctness() {
        let block_size = 32;

        // Create test data
        let scales = vec![1.0f32, 0.5f32, 2.0f32];
        let n_blocks = scales.len();
        let output_size = n_blocks * block_size;

        // Create quantized data
        let mut quants = Vec::new();
        for _ in 0..n_blocks {
            for _ in 0..(block_size + 1) / 2 {
                quants.push(0x88); // Pattern: 8, 8, 8, 8, ...
            }
        }

        let mut output = vec![0.0f32; output_size];

        dequantize_q4_0_simd(&quants, &scales, block_size, &mut output).unwrap();

        // Verify first block (scale = 1.0, quants = 8)
        for i in 0..block_size {
            assert_eq!(output[i], -8.0);
        }

        // Verify second block (scale = 0.5, quants = 8)
        for i in block_size..2 * block_size {
            assert_eq!(output[i], -4.0);
        }
    }
}
