//! Tensor operations implementation
//!
//! This module provides a comprehensive set of tensor operations optimized
//! for performance with SIMD support where applicable.

use crate::tensor::{Tensor, TensorType, Shape};
use crate::error::{Error, Result};

/// Trait for unary tensor operations
pub trait UnaryOp {
    fn apply(&self, input: &Tensor) -> Result<Tensor>;
}

/// Trait for binary tensor operations
pub trait BinaryOp {
    fn apply(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
}

/// Addition operation
#[derive(Debug, Clone)]
pub struct Add;

impl BinaryOp for Add {
    fn apply(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        // Check shapes are compatible
        if !lhs.shape().is_broadcastable_to(rhs.shape()) {
            return Err(Error::dimension_mismatch(
                format!("Cannot add {:?} and {:?}", lhs.shape(), rhs.shape())
            ));
        }

        // For now, only implement same-shape addition
        if lhs.shape() != rhs.shape() {
            return Err(Error::Unsupported(
                "Broadcasting not yet implemented".to_string()
            ));
        }

        match (lhs.dtype(), rhs.dtype()) {
            (TensorType::F32, TensorType::F32) => {
                let lhs_data = lhs.as_f32_slice()?;
                let rhs_data = rhs.as_f32_slice()?;

                let result: Vec<f32> = lhs_data.iter()
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
            _ => Err(Error::Unsupported(
                format!("Add not implemented for {} + {}", lhs.dtype(), rhs.dtype())
            ))
        }
    }
}

/// Matrix multiplication
#[derive(Debug, Clone)]
pub struct MatMul;

impl BinaryOp for MatMul {
    fn apply(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        if !lhs.is_matrix() || !rhs.is_matrix() {
            return Err(Error::tensor("MatMul requires 2D tensors"));
        }

        let lhs_shape = lhs.shape().dims();
        let rhs_shape = rhs.shape().dims();

        // Check dimensions: (M, K) @ (K, N) = (M, N)
        if lhs_shape[1] != rhs_shape[0] {
            return Err(Error::dimension_mismatch(
                format!("Cannot multiply {:?} by {:?}", lhs_shape, rhs_shape)
            ));
        }

        let m = lhs_shape[0];
        let k = lhs_shape[1];
        let n = rhs_shape[1];

        match (lhs.dtype(), rhs.dtype()) {
            (TensorType::F32, TensorType::F32) => {
                let lhs_data = lhs.as_f32_slice()?;
                let rhs_data = rhs.as_f32_slice()?;

                let mut result = vec![0.0f32; m * n];

                // Simple matrix multiplication (can be optimized with SIMD)
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            sum += unsafe {
                                *lhs_data.get_unchecked(i * k + l) *
                                *rhs_data.get_unchecked(l * n + j)
                            };
                        }
                        result[i * n + j] = sum;
                    }
                }

                Tensor::new(
                    None,
                    TensorType::F32,
                    Shape::matrix(m, n),
                    crate::tensor::TensorData::F32(result),
                )
            }
            _ => Err(Error::Unsupported(
                format!("MatMul not implemented for {} x {}", lhs.dtype(), rhs.dtype())
            ))
        }
    }
}

/// ReLU activation function
#[derive(Debug, Clone)]
pub struct Relu;

impl UnaryOp for Relu {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        match input.dtype() {
            TensorType::F32 => {
                let data = input.as_f32_slice()?;
                let result: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();

                Tensor::new(
                    input.name().map(|s| s.to_string()),
                    TensorType::F32,
                    input.shape().clone(),
                    crate::tensor::TensorData::F32(result),
                )
            }
            _ => Err(Error::Unsupported(
                format!("ReLU not implemented for {}", input.dtype())
            ))
        }
    }
}

/// Gelu activation function (Gaussian Error Linear Unit)
#[derive(Debug, Clone)]
pub struct Gelu;

impl UnaryOp for Gelu {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        match input.dtype() {
            TensorType::F32 => {
                let data = input.as_f32_slice()?;
                const GELU_CONST: f32 = 0.044715;

                let result: Vec<f32> = data.iter().map(|&x| {
                    0.5 * x * (1.0 + (x * GELU_CONST * x * x).tanh())
                }).collect();

                Tensor::new(
                    input.name().map(|s| s.to_string()),
                    TensorType::F32,
                    input.shape().clone(),
                    crate::tensor::TensorData::F32(result),
                )
            }
            _ => Err(Error::Unsupported(
                format!("GELU not implemented for {}", input.dtype())
            ))
        }
    }
}

/// SiLU activation function (Swish)
#[derive(Debug, Clone)]
pub struct Silu;

impl UnaryOp for Silu {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        match input.dtype() {
            TensorType::F32 => {
                let data = input.as_f32_slice()?;
                let result: Vec<f32> = data.iter().map(|&x| {
                    x / (1.0 + (-x).exp())
                }).collect();

                Tensor::new(
                    input.name().map(|s| s.to_string()),
                    TensorType::F32,
                    input.shape().clone(),
                    crate::tensor::TensorData::F32(result),
                )
            }
            _ => Err(Error::Unsupported(
                format!("SiLU not implemented for {}", input.dtype())
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorData;

    #[test]
    fn test_add() {
        let a = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(2, 2),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        ).unwrap();

        let b = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(2, 2),
            TensorData::F32(vec![5.0, 6.0, 7.0, 8.0]),
        ).unwrap();

        let op = Add;
        let result = op.apply(&a, &b).unwrap();

        assert_eq!(result.as_f32_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(2, 3),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ).unwrap();

        let b = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(3, 2),
            TensorData::F32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ).unwrap();

        let op = MatMul;
        let result = op.apply(&a, &b).unwrap();

        // [[1,2,3], [4,5,6]] @ [[7,8], [9,10], [11,12]]
        // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(result.as_f32_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::new(
            None,
            TensorType::F32,
            Shape::vector(4),
            TensorData::F32(vec![-1.0, 0.0, 1.0, 2.0]),
        ).unwrap();

        let op = Relu;
        let result = op.apply(&a).unwrap();

        assert_eq!(result.as_f32_slice().unwrap(), &[0.0, 0.0, 1.0, 2.0]);
    }
}
