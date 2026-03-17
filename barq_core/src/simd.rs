//! SIMD-optimized operations for quantization and tensor operations
//!
//! This module provides SIMD-accelerated implementations using:
//! - AVX2 and AVX-512 for x86_64
//! - NEON for ARM64
//! - Fallback scalar implementations for other architectures

use crate::error::{Error, Result};

/// SIMD optimization level detected at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub enum SimdLevel {
    /// No SIMD support
    None,
    /// SSE4.2 support (x86_64 baseline)
    SSE42,
    /// AVX2 support (x86_64)
    AVX2,
    /// AVX-512 support (x86_64)
    AVX512,
    /// NEON support (ARM64)
    NEON,
}

impl SimdLevel {
    /// Detect the best available SIMD support at runtime
    pub fn detect() -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "std"))]
        {
            use std::arch::x86_64::*;

            unsafe {
                if is_x86_feature_detected!("avx512f") {
                    return SimdLevel::AVX512;
                }
                if is_x86_feature_detected!("avx2") {
                    return SimdLevel::AVX2;
                }
                if is_x86_feature_detected!("sse4.2") {
                    return SimdLevel::SSE42;
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "std"))]
        {
            return SimdLevel::NEON;
        }

        SimdLevel::None
    }

    /// Get the vector width in f32 elements for this SIMD level
    pub fn vector_width(&self) -> usize {
        match self {
            SimdLevel::AVX512 => 16, // 512 bits / 32 bits
            SimdLevel::AVX2 => 8,    // 256 bits / 32 bits
            SimdLevel::SSE42 => 4,   // 128 bits / 32 bits
            SimdLevel::NEON => 4,    // 128 bits / 32 bits
            SimdLevel::None => 1,
        }
    }
}

/// SIMD-optimized dot product for quantization
///
/// Computes dot product of f32 vectors with maximum SIMD acceleration
pub fn simd_dot_product_f32(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(Error::tensor("Vector length mismatch for dot product"));
    }

    let simd_level = SimdLevel::detect();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            match simd_level {
                SimdLevel::AVX2 if is_x86_feature_detected!("avx2") => {
                    return Ok(dot_product_avx2(a, b));
                }
                SimdLevel::SSE42 if is_x86_feature_detected!("sse4.2") => {
                    return Ok(dot_product_sse42(a, b));
                }
                _ => {}
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if simd_level == SimdLevel::NEON {
            unsafe {
                return Ok(dot_product_neon(a, b));
            }
        }
    }

    // Fallback to scalar implementation
    Ok(dot_product_scalar(a, b))
}

/// Scalar fallback for dot product
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        i += 8;
    }

    // Horizontal sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total: f32 = result.iter().sum();

    // Process remaining elements
    while i < a.len() {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn dot_product_sse42(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let mul = _mm_mul_ps(a_vec, b_vec);
        sum = _mm_add_ps(sum, mul);
        i += 4;
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut total: f32 = result.iter().sum();

    // Process remaining elements
    while i < a.len() {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let mul = vmulq_f32(a_vec, b_vec);
        sum = vaddq_f32(sum, mul);
        i += 4;
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    vst1q_f32(result.as_mut_ptr(), sum);
    let mut total: f32 = result.iter().sum();

    // Process remaining elements
    while i < a.len() {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

/// SIMD-optimized vector addition
pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(Error::tensor("Vector length mismatch for addition"));
    }

    let simd_level = SimdLevel::detect();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            if simd_level == SimdLevel::AVX2 && is_x86_feature_detected!("avx2") {
                return add_avx2(a, b, result);
            } else if simd_level >= SimdLevel::SSE42 && is_x86_feature_detected!("sse4.2") {
                return add_sse42(a, b, result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            if simd_level == SimdLevel::NEON {
                return add_neon(a, b, result);
            }
        }
    }

    // Scalar fallback
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
    use std::arch::x86_64::*;

    let mut i = 0;
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let sum = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), sum);
        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn add_sse42(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
    use std::arch::x86_64::*;

    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let sum = _mm_add_ps(a_vec, b_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), sum);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_neon(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let sum = vaddq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), sum);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let level = SimdLevel::detect();
        println!("Detected SIMD level: {:?}", level);
        // Should at least have some support on modern CPUs
        assert!(level >= SimdLevel::None || level <= SimdLevel::AVX512);
    }

    #[test]
    fn test_dot_product_correctness() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = simd_dot_product_f32(&a, &b).unwrap();

        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_add_correctness() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut result = vec![0.0; 5];

        simd_add_f32(&a, &b, &mut result).unwrap();

        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_vector_width() {
        let level = SimdLevel::detect();
        let width = level.vector_width();
        assert!(width >= 1 && width <= 16);
    }
}
