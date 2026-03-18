//! Optimized matrix operations
//!
//! Provides parallelized matrix multiplication using rayon for CPU acceleration.
//! Future versions will add BLAS support for additional performance.

use crate::error::{Error, Result};
use rayon::prelude::*;

/// Matrix multiplication: C = A * B
///
/// # Arguments
/// * `a` - Matrix A of shape (m, k) in row-major order
/// * `b` - Matrix B of shape (k, n) in row-major order
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A and rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
/// Matrix C of shape (m, n) in row-major order
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    if a.len() != m * k {
        return Err(Error::tensor(format!(
            "Matrix A has {} elements but expected {} (m={}) * (k={})",
            a.len(),
            m * k,
            m,
            k
        )));
    }
    if b.len() != k * n {
        return Err(Error::tensor(format!(
            "Matrix B has {} elements but expected {} (k={}) * (n={})",
            b.len(),
            k * n,
            k,
            n
        )));
    }

    // Parallel matrix multiplication
    let mut c = vec![0.0f32; m * n];

    // Parallelize over rows of A/C
    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += unsafe { *a.get_unchecked(i * k + kk) * b.get_unchecked(kk * n + j) };
            }
            row[j] = sum;
        }
    });

    Ok(c)
}

/// Matrix-vector multiplication: y = A * x
///
/// # Arguments
/// * `a` - Matrix A of shape (m, n) in row-major order
/// * `x` - Vector x of length n
/// * `m` - Number of rows in A
/// * `n` - Number of columns in A and length of x
///
/// # Returns
/// Vector y of length m
pub fn gemv_f32(a: &[f32], x: &[f32], m: usize, n: usize) -> Result<Vec<f32>> {
    if a.len() != m * n {
        return Err(Error::tensor(format!(
            "Matrix A has {} elements but expected {} (m={}) * (n={})",
            a.len(),
            m * n,
            m,
            n
        )));
    }
    if x.len() != n {
        return Err(Error::tensor(format!(
            "Vector x has {} elements but expected {}",
            x.len(),
            n
        )));
    }

    let mut y = vec![0.0f32; m];

    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        y[i] = sum;
    }

    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_f32_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix

        let result = gemm_f32(&a, &b, 2, 2, 2).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert_eq!(result[0], 19.0);
        assert_eq!(result[1], 22.0);
        assert_eq!(result[2], 43.0);
        assert_eq!(result[3], 50.0);
    }

    #[test]
    fn test_gemm_f32_rectangular() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 matrix

        let result = gemm_f32(&a, &b, 2, 3, 2).unwrap();

        // Expected: [[58, 64], [139, 154]]
        assert!((result[0] - 58.0).abs() < 1e-5);
        assert!((result[1] - 64.0).abs() < 1e-5);
        assert!((result[2] - 139.0).abs() < 1e-5);
        assert!((result[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let x = vec![5.0, 6.0]; // vector of length 2

        let result = gemv_f32(&a, &x, 2, 2).unwrap();

        // Expected: [17, 39]
        assert_eq!(result[0], 17.0);
        assert_eq!(result[1], 39.0);
    }

    #[test]
    fn test_invalid_dimensions() {
        let a = vec![1.0, 2.0, 3.0]; // Wrong size for 2x2
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let result = gemm_f32(&a, &b, 2, 2, 2);
        assert!(result.is_err());
    }
}
