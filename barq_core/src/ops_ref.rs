//! Reference implementation of tensor operations
//!
//! This module provides reference implementations that prioritize correctness
//! over performance. Used for testing and fallback.

use crate::error::{Error, Result};
use crate::tensor::{Shape, Tensor, TensorType};

/// Reference implementation of addition
pub fn add_ref(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.shape() != rhs.shape() {
        return Err(Error::dimension_mismatch(format!(
            "Cannot add {:?} and {:?}",
            lhs.shape(),
            rhs.shape()
        )));
    }

    if lhs.dtype() != rhs.dtype() {
        return Err(Error::type_mismatch(lhs.dtype().name(), rhs.dtype().name()));
    }

    match lhs.dtype() {
        TensorType::F32 => {
            let lhs_data = lhs.as_f32_slice()?;
            let rhs_data = rhs.as_f32_slice()?;
            let result: Vec<f32> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a + b)
                .collect();
            Tensor::new(
                None,
                TensorType::F32,
                lhs.shape().clone(),
                crate::tensor::TensorData::F32(result),
            )
        }
        _ => Err(Error::Unsupported(format!(
            "Add not implemented for {}",
            lhs.dtype()
        ))),
    }
}

/// Reference implementation of matrix multiplication
pub fn matmul_ref(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if !lhs.is_matrix() || !rhs.is_matrix() {
        return Err(Error::tensor("MatMul requires 2D tensors"));
    }

    let lhs_shape = lhs.shape().dims();
    let rhs_shape = rhs.shape().dims();

    if lhs_shape[1] != rhs_shape[0] {
        return Err(Error::dimension_mismatch(format!(
            "Cannot multiply {:?} by {:?}",
            lhs_shape, rhs_shape
        )));
    }

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    match (lhs.dtype(), rhs.dtype()) {
        (TensorType::F32, TensorType::F32) => {
            let lhs_data = lhs.as_f32_slice()?;
            let rhs_data = rhs.as_f32_slice()?;

            let mut result = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        result[i * n + j] += unsafe {
                            *lhs_data.get_unchecked(i * k + l) * *rhs_data.get_unchecked(l * n + j)
                        };
                    }
                }
            }

            Tensor::new(
                None,
                TensorType::F32,
                Shape::matrix(m, n),
                crate::tensor::TensorData::F32(result),
            )
        }
        _ => Err(Error::Unsupported(format!(
            "MatMul not implemented for {} x {}",
            lhs.dtype(),
            rhs.dtype()
        ))),
    }
}
