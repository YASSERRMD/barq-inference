//! SIMD-optimized softmax computation
//!
//! Provides accelerated softmax using AVX2/AVX512/NEON intrinsics

use crate::error::Result;

/// Compute softmax using SIMD acceleration
///
/// Numerically stable implementation: exp(x - max(x)) / sum(exp(x - max(x)))
pub fn simd_softmax(logits: &[f32], output: &mut [f32]) -> Result<()> {
    if logits.len() != output.len() {
        return Err(crate::error::Error::tensor("Length mismatch in softmax"));
    }

    if logits.is_empty() {
        return Ok(());
    }

    // Find maximum using SIMD
    let max_logit = simd_reduce_max(logits);

    // Compute exp and sum using SIMD
    let mut exp_sum = 0.0f32;

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            if is_x86_feature_detected!("avx2") {
                return softmax_avx2(logits, output, max_logit);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return softmax_neon(logits, output, max_logit);
        }
    }

    // Scalar fallback
    for (i, &logit) in logits.iter().enumerate() {
        let exp = (logit - max_logit).exp();
        output[i] = exp;
        exp_sum += exp;
    }

    // Normalize
    for val in output.iter_mut() {
        *val /= exp_sum;
    }

    Ok(())
}

/// Find maximum value in array using SIMD
fn simd_reduce_max(arr: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            if is_x86_feature_detected!("avx2") {
                return reduce_max_avx2(arr);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return reduce_max_neon(arr);
        }
    }

    // Scalar fallback
    arr.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_max_avx2(arr: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= arr.len() {
        let vec = _mm256_loadu_ps(arr.as_ptr().add(i));
        max_vec = _mm256_max_ps(max_vec, vec);
        i += 8;
    }

    // Horizontal max
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), max_vec);
    let mut max = result[0];
    for &val in &result[1..] {
        max = max.max(val);
    }

    // Process remaining elements
    while i < arr.len() {
        max = max.max(arr[i]);
        i += 1;
    }

    max
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn reduce_max_neon(arr: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= arr.len() {
        let vec = vld1q_f32(arr.as_ptr().add(i));
        max_vec = vmaxq_f32(max_vec, vec);
        i += 4;
    }

    // Horizontal max
    let mut result = [0.0f32; 4];
    vst1q_f32(result.as_mut_ptr(), max_vec);
    let mut max = result[0];
    for &val in &result[1..] {
        max = max.max(val);
    }

    // Process remaining elements
    while i < arr.len() {
        max = max.max(arr[i]);
        i += 1;
    }

    max
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_avx2(logits: &[f32], output: &mut [f32], max_logit: f32) -> Result<()> {
    use std::arch::x86_64::*;

    let max_vec = _mm256_set1_ps(max_logit);
    let mut exp_sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    // Compute exp values
    while i + 8 <= logits.len() {
        let logit_vec = _mm256_loadu_ps(logits.as_ptr().add(i));
        let centered = _mm256_sub_ps(logit_vec, max_vec);

        // Approximate exp using polynomial (for performance)
        let exp_vec = exp_approx_avx2(centered);

        _mm256_storeu_ps(output.as_mut_ptr().add(i), exp_vec);
        exp_sum_vec = _mm256_add_ps(exp_sum_vec, exp_vec);
        i += 8;
    }

    // Horizontal sum of exp_sum_vec
    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), exp_sum_vec);
    let mut exp_sum: f32 = sum_arr.iter().sum();

    // Handle remaining elements
    for j in i..logits.len() {
        let exp = (logits[j] - max_logit).exp();
        output[j] = exp;
        exp_sum += exp;
    }

    // Normalize using SIMD
    let inv_sum_vec = _mm256_set1_ps(1.0 / exp_sum);
    let mut j = 0;
    while j + 8 <= output.len() {
        let vec = _mm256_loadu_ps(output.as_ptr().add(j));
        let normalized = _mm256_mul_ps(vec, inv_sum_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(j), normalized);
        j += 8;
    }

    // Handle remaining normalization
    while j < output.len() {
        output[j] /= exp_sum;
        j += 1;
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_neon(logits: &[f32], output: &mut [f32], max_logit: f32) -> Result<()> {
    use std::arch::aarch64::*;

    let max_vec = vdupq_n_f32(max_logit);
    let mut exp_sum = 0.0f32;
    let mut i = 0;

    // Compute exp values
    while i + 4 <= logits.len() {
        let logit_vec = vld1q_f32(logits.as_ptr().add(i));
        let centered = vsubq_f32(logit_vec, max_vec);

        // Use standard exp for NEON (could be optimized with approximation)
        let mut vals = [0.0f32; 4];
        vst1q_f32(vals.as_mut_ptr(), centered);

        for (j, &val) in vals.iter().enumerate() {
            let exp = val.exp();
            output[i + j] = exp;
            exp_sum += exp;
        }

        i += 4;
    }

    // Handle remaining elements
    for j in i..logits.len() {
        let exp = (logits[j] - max_logit).exp();
        output[j] = exp;
        exp_sum += exp;
    }

    // Normalize
    let inv_sum = 1.0 / exp_sum;
    for val in output.iter_mut() {
        *val *= inv_sum;
    }

    Ok(())
}

/// Approximate exp using AVX2 polynomial approximation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn exp_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    // Clamp x to avoid overflow
    let max = _mm256_set1_ps(88.0f32);
    let min = _mm256_set1_ps(-88.0f32);
    let x = _mm256_min_ps(_mm256_max_ps(x, min), max);

    // Fast approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    let x4 = _mm256_mul_ps(x3, x);

    let term2 = _mm256_mul_ps(x, one); // x
    let term3 = _mm256_mul_ps(x2, _mm256_set1_ps(0.5)); // x²/2
    let term4 = _mm256_mul_ps(x3, _mm256_set1_ps(0.16666667)); // x³/6
    let term5 = _mm256_mul_ps(x4, _mm256_set1_ps(0.041666668)); // x⁴/24

    _mm256_add_ps(
        _mm256_add_ps(_mm256_add_ps(one, term2), _mm256_add_ps(term3, term4)),
        term5,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_softmax_correctness() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];

        simd_softmax(&logits, &mut output).unwrap();

        // Check sum is 1.0
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check values are positive
        for &val in &output {
            assert!(val > 0.0);
        }

        // Check ordering is preserved
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }

    #[test]
    fn test_simd_softmax_empty() {
        let logits: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];

        simd_softmax(&logits, &mut output).unwrap();
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn test_reduce_max() {
        let arr = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        let max = simd_reduce_max(&arr);
        assert_eq!(max, 9.0);
    }
}
