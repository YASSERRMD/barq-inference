//! Optimized matrix operations
//!
//! Provides hardware-accelerated matrix multiplication using:
//! 1. Apple Accelerate framework (cBLAS) on macOS - PRIMARY
//! 2. Metal GPU acceleration on Apple Silicon - FALLBACK 1
//! 3. CPU parallelization using rayon - FALLBACK 2

use crate::accelerate_blas;
use crate::error::{Error, Result};
use crate::metal_blas::MetalBlas;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::OnceLock;

// Global Metal BLAS instance (lazy initialization)
static METAL_BLAS: OnceLock<Option<Arc<MetalBlas>>> = OnceLock::new();

/// Get or initialize Metal BLAS instance
fn get_metal_blas() -> Option<&'static Arc<MetalBlas>> {
    METAL_BLAS
        .get_or_init(|| {
            #[cfg(feature = "metal")]
            {
                match MetalBlas::new() {
                    Ok(blas) => {
                        eprintln!("Metal BLAS initialized successfully");
                        Some(Arc::new(blas))
                    }
                    Err(e) => {
                        eprintln!("Failed to initialize Metal BLAS: {}", e);
                        eprintln!("Falling back to CPU parallelization");
                        None
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                None
            }
        })
        .as_ref()
}

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
///
/// # Backend Priority
/// 1. Apple Accelerate (cBLAS) on macOS - hardware-optimized SIMD
/// 2. Metal GPU on Apple Silicon - GPU acceleration
/// 3. Rayon CPU parallelization - multi-threaded fallback
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

    // Try Apple Accelerate (cBLAS) first - PRIMARY BACKEND on macOS
    #[cfg(target_os = "macos")]
    {
        if accelerate_blas::is_available() {
            match accelerate_blas::sgemm(a, b, m, k, n) {
                Ok(result) => {
                    // Only print on first few calls to avoid spam
                    static FIRST_CALL: std::sync::Once = std::sync::Once::new();
                    FIRST_CALL.call_once(|| {
                        eprintln!("Using Apple Accelerate cBLAS for matrix multiplication");
                    });
                    return Ok(result);
                }
                Err(e) => {
                    // Only print once
                    static FIRST_FAIL: std::sync::Once = std::sync::Once::new();
                    FIRST_FAIL.call_once(|| {
                        eprintln!("Accelerate GEMM failed: {}, falling back to Metal", e);
                    });
                }
            }
        }
    }

    // Try Metal GPU acceleration next
    if let Some(metal_blas) = get_metal_blas() {
        match metal_blas.gemm(a, b, m, k, n) {
            Ok(result) => {
                static FIRST_METAL: std::sync::Once = std::sync::Once::new();
                FIRST_METAL.call_once(|| {
                    eprintln!("Using Metal GPU for matrix multiplication");
                });
                return Ok(result);
            }
            Err(e) => {
                static FIRST_METAL_FAIL: std::sync::Once = std::sync::Once::new();
                FIRST_METAL_FAIL.call_once(|| {
                    eprintln!("Metal GEMM failed: {}, falling back to CPU", e);
                });
            }
        }
    }

    // Final fallback to CPU parallelization with rayon
    static FIRST_CPU: std::sync::Once = std::sync::Once::new();
    FIRST_CPU.call_once(|| {
        eprintln!("Using CPU parallelization (rayon) for matrix multiplication");
    });

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
///
/// # Backend Priority
/// 1. Apple Accelerate (cBLAS) on macOS - hardware-optimized SIMD
/// 2. Rayon CPU parallelization - multi-threaded fallback
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

    // Try Apple Accelerate (cBLAS) first on macOS
    #[cfg(target_os = "macos")]
    {
        if accelerate_blas::is_available() {
            match accelerate_blas::sgemv(a, x, m, n) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("Accelerate GEMV failed: {}, falling back to CPU", e);
                }
            }
        }
    }

    // Fallback to CPU
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
