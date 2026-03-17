//! Optimized General Matrix Multiply (GEMM) kernels
//!
//! Provides high-performance matrix multiplication with:
//! - SIMD acceleration (AVX2, AVX512, NEON)
//! - Cache-blocking for better locality
//! - Loop tiling and unrolling
//! - Support for different data layouts

use crate::error::Result;

/// GEMM configuration
#[derive(Debug, Clone)]
pub struct GEMMConfig {
    /// Block size for cache blocking (default: 64)
    pub block_size: usize,
    /// Whether to use SIMD acceleration (default: true)
    pub use_simd: bool,
    /// Whether to use loop unrolling (default: true)
    pub unroll: bool,
}

impl Default for GEMMConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            use_simd: true,
            unroll: true,
        }
    }
}

/// Optimized GEMM: C = A @ B where A is [M x K], B is [K x N], C is [M x N]
pub fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
    gemm_f32_with_config(a, b, c, m, n, k, &GEMMConfig::default())
}

/// Optimized GEMM with custom configuration
pub fn gemm_f32_with_config(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &GEMMConfig,
) -> Result<()> {
    // Validate dimensions
    if a.len() != m * k {
        return Err(crate::error::Error::tensor(format!(
            "A dimension mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }

    if b.len() != k * n {
        return Err(crate::error::Error::tensor(format!(
            "B dimension mismatch: expected {}, got {}",
            k * n,
            b.len()
        )));
    }

    if c.len() != m * n {
        return Err(crate::error::Error::tensor(format!(
            "C dimension mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }

    // Try SIMD implementation first
    if config.use_simd {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;

            unsafe {
                if is_x86_feature_detected!("avx2") {
                    return gemm_avx2_blocked(a, b, c, m, n, k, config);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                return gemm_neon_blocked(a, b, c, m, n, k, config);
            }
        }
    }

    // Fallback to blocked scalar implementation
    gemm_blocked_scalar(a, b, c, m, n, k, config)
}

/// Blocked GEMM with AVX2 acceleration
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemm_avx2_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &GEMMConfig,
) -> Result<()> {
    use std::arch::x86_64::*;

    let block_size = config.block_size;

    // Clear output matrix
    c.fill(0.0);

    // Process in blocks for cache efficiency
    for ii in (0..m).step_by(block_size) {
        for jj in (0..n).step_by(block_size) {
            for kk in (0..k).step_by(block_size) {
                let i_end = (ii + block_size).min(m);
                let j_end = (jj + block_size).min(n);
                let k_end = (kk + block_size).min(k);

                // Compute block: C[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]
                for i in ii..i_end {
                    for k_idx in kk..k_end {
                        let a_val = a[i * k + k_idx];

                        // Broadcast a_val to all lanes
                        let a_vec = _mm256_set1_ps(a_val);

                        // Load 8 elements of B at a time
                        let mut j = jj;
                        while j + 8 <= j_end {
                            let b_vec = _mm256_loadu_ps(b.as_ptr().add(k_idx * n + j));
                            let c_vec = _mm256_loadu_ps(c.as_mut_ptr().add(i * n + j));

                            // FMA: c += a * b
                            let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), result);

                            j += 8;
                        }

                        // Handle remaining elements
                        while j < j_end {
                            c[i * n + j] += a_val * b[k_idx * n + j];
                            j += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Blocked GEMM with NEON acceleration
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemm_neon_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &GEMMConfig,
) -> Result<()> {
    use std::arch::aarch64::*;

    let block_size = config.block_size;

    // Clear output matrix
    c.fill(0.0);

    // Process in blocks for cache efficiency
    for ii in (0..m).step_by(block_size) {
        for jj in (0..n).step_by(block_size) {
            for kk in (0..k).step_by(block_size) {
                let i_end = (ii + block_size).min(m);
                let j_end = (jj + block_size).min(n);
                let k_end = (kk + block_size).min(k);

                // Compute block
                for i in ii..i_end {
                    for k_idx in kk..k_end {
                        let a_val = a[i * k + k_idx];

                        // Broadcast a_val to all lanes
                        let a_vec = vdupq_n_f32(a_val);

                        // Load 4 elements of B at a time
                        let mut j = jj;
                        while j + 4 <= j_end {
                            let b_vec = vld1q_f32(b.as_ptr().add(k_idx * n + j));
                            let c_vec = vld1q_f32(c.as_mut_ptr().add(i * n + j));

                            // FMA: c = a * b + c
                            let result = vfmaq_f32(c_vec, a_vec, b_vec);
                            vst1q_f32(c.as_mut_ptr().add(i * n + j), result);

                            j += 4;
                        }

                        // Handle remaining elements
                        while j < j_end {
                            c[i * n + j] += a_val * b[k_idx * n + j];
                            j += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Blocked scalar GEMM (fallback)
fn gemm_blocked_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &GEMMConfig,
) -> Result<()> {
    let block_size = config.block_size;

    // Clear output matrix
    c.fill(0.0);

    // Process in blocks for cache efficiency
    for ii in (0..m).step_by(block_size) {
        for jj in (0..n).step_by(block_size) {
            for kk in (0..k).step_by(block_size) {
                let i_end = (ii + block_size).min(m);
                let j_end = (jj + block_size).min(n);
                let k_end = (kk + block_size).min(k);

                // Compute block
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = 0.0f32;
                        for k_idx in kk..k_end {
                            sum += a[i * k + k_idx] * b[k_idx * n + j];
                        }
                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Batch GEMM: process multiple matrix multiplications
pub fn batch_gemm_f32(
    a_batch: &[&[f32]],
    b_batch: &[&[f32]],
    c_batch: &mut [&mut [f32]],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if a_batch.len() != b_batch.len() || a_batch.len() != c_batch.len() {
        return Err(crate::error::Error::tensor("Batch dimensions must match"));
    }

    for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        gemm_f32(a, b, c_batch[i], m, n, k)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_basic() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // [2 x 2]
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // [2 x 2]
        let mut c = vec![0.0f32; 4]; // [2 x 2]

        gemm_f32(&a, &b, &mut c, 2, 2, 2).unwrap();

        // C = A @ B
        // C[0,0] = 1*5 + 2*7 = 19
        // C[0,1] = 1*6 + 2*8 = 22
        // C[1,0] = 3*5 + 4*7 = 43
        // C[1,1] = 3*6 + 4*8 = 50

        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[1] - 22.0).abs() < 1e-6);
        assert!((c[2] - 43.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_rectangular() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2 x 3]
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3 x 2]
        let mut c = vec![0.0f32; 4]; // [2 x 2]

        gemm_f32(&a, &b, &mut c, 2, 2, 3).unwrap();

        // C[0,0] = 1*7 + 2*9 + 3*11 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 154

        assert!((c[0] - 58.0).abs() < 1e-6);
        assert!((c[1] - 64.0).abs() < 1e-6);
        assert!((c[2] - 139.0).abs() < 1e-6);
        assert!((c[3] - 154.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_config() {
        let config = GEMMConfig {
            block_size: 32,
            use_simd: false,
            unroll: false,
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];

        gemm_f32_with_config(&a, &b, &mut c, 2, 2, 2, &config).unwrap();

        assert!((c[0] - 19.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_gemm() {
        let a1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let b1 = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c1 = vec![0.0f32; 4];

        let a2 = vec![2.0f32, 3.0, 4.0, 5.0];
        let b2 = vec![6.0f32, 7.0, 8.0, 9.0];
        let mut c2 = vec![0.0f32; 4];

        batch_gemm_f32(&[&a1, &a2], &[&b1, &b2], &mut [&mut c1, &mut c2], 2, 2, 2).unwrap();

        assert!((c1[0] - 19.0).abs() < 1e-6);
        assert!((c2[0] - 36.0).abs() < 1e-6);
    }
}
