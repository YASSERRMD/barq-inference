//! Apple Accelerate framework BLAS bindings
//!
//! Provides optimized BLAS operations using direct FFI bindings to
//! Apple's Accelerate framework cBLAS on macOS.

use crate::error::{Error, Result};
use std::os::raw::{c_char, c_int};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// cblas_sgemm - single-precision general matrix multiply
    /// Try with scalars passed by value instead of by reference
    fn cblas_sgemm(
        layout: c_int, // Matrix layout: 101=RowMajor, 102=ColMajor
        transa: c_int, // Transpose A: 111=NoTrans, 112=Trans, 113=ConjTrans
        transb: c_int, // Transpose B
        m: c_int,      // Rows of op(A) and C
        n: c_int,      // Columns of op(B) and C
        k: c_int,      // Columns of op(A) and rows of op(B)
        alpha: f32,    // Scalar (pass by value)
        a: *const f32, // Matrix A
        lda: c_int,    // Leading dimension of A
        b: *const f32, // Matrix B
        ldb: c_int,    // Leading dimension of B
        beta: f32,     // Scalar (pass by value)
        c: *mut f32,   // Matrix C
        ldc: c_int,    // Leading dimension of C
    );

    /// cblas_sgemv - single-precision general matrix-vector multiply
    fn cblas_sgemv(
        layout: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        x: *const f32,
        incx: c_int,
        beta: f32,
        y: *mut f32,
        incy: c_int,
    );
}

/// Check if Accelerate is available
#[cfg(target_os = "macos")]
pub fn is_available() -> bool {
    true // Accelerate is always available on macOS
}

/// Matrix multiplication using Apple Accelerate's cBLAS
#[cfg(target_os = "macos")]
pub fn sgemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let a_len = m * k;
    let b_len = k * n;
    let c_len = m * n;

    if a.len() != a_len {
        return Err(Error::tensor(format!(
            "Matrix A size mismatch: expected {}, got {}",
            a_len,
            a.len()
        )));
    }
    if b.len() != b_len {
        return Err(Error::tensor(format!(
            "Matrix B size mismatch: expected {}, got {}",
            b_len,
            b.len()
        )));
    }

    let mut c = vec![0.0f32; c_len];

    // Simple approach: Use row-major, straightforward parameters
    // Pass scalars by value (changed FFI signature)
    unsafe {
        cblas_sgemm(
            101, // CblasRowMajor
            111, // CblasNoTrans
            111, // CblasNoTrans
            m as c_int,
            n as c_int,
            k as c_int,
            1.0f32, // alpha (by value)
            a.as_ptr(),
            k as c_int, // lda = columns of A
            b.as_ptr(),
            n as c_int, // ldb = columns of B
            0.0f32,     // beta (by value)
            c.as_mut_ptr(),
            n as c_int, // ldc = columns of C
        );
    }

    Ok(c)
}

/// Matrix-vector multiplication using Apple Accelerate (via sgemm with m=1)
#[cfg(target_os = "macos")]
pub fn sgemv(a: &[f32], x: &[f32], m: usize, n: usize) -> Result<Vec<f32>> {
    // A is (m, n), x is (n,) → result is (m,)
    // Treat x as a column matrix (n, 1) and call sgemm
    sgemm(a, x, m, n, 1)
}

// Stub implementations for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub fn is_available() -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn sgemm(_a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Result<Vec<f32>> {
    Err(Error::Unsupported(
        "Apple Accelerate is only available on macOS".to_string(),
    ))
}

#[cfg(not(target_os = "macos"))]
pub fn sgemv(_a: &[f32], _x: &[f32], _m: usize, _n: usize) -> Result<Vec<f32>> {
    Err(Error::Unsupported(
        "Apple Accelerate is only available on macOS".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_accelerate_available() {
        assert!(is_available());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_sgemm_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix

        let result = sgemm(&a, &b, 2, 2, 2).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert!((result[0] - 19.0).abs() < 1e-5);
        assert!((result[1] - 22.0).abs() < 1e-5);
        assert!((result[2] - 43.0).abs() < 1e-5);
        assert!((result[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_sgemm_rectangular() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 matrix

        let result = sgemm(&a, &b, 2, 3, 2).unwrap();

        // Expected: [[58, 64], [139, 154]]
        assert!((result[0] - 58.0).abs() < 1e-5);
        assert!((result[1] - 64.0).abs() < 1e-5);
        assert!((result[2] - 139.0).abs() < 1e-5);
        assert!((result[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_sgemv() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let x = vec![5.0, 6.0]; // vector of length 2

        let result = sgemv(&a, &x, 2, 2).unwrap();

        // Expected: [17, 39]
        assert!((result[0] - 17.0).abs() < 1e-5);
        assert!((result[1] - 39.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_invalid_dimensions() {
        let a = vec![1.0, 2.0, 3.0]; // Wrong size for 2x2
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let result = sgemm(&a, &b, 2, 2, 2);
        assert!(result.is_err());
    }
}
