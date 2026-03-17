//! Tensor operations abstraction

use barq_core::error::{Error, Result};
use barq_core::ops::{BinaryOp, UnaryOp};
use barq_core::tensor::Tensor;

/// Tensor operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorOp {
    /// Addition
    Add,
    /// Matrix multiplication
    MatMul,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// ReLU activation
    Relu,
    /// GELU activation
    Gelu,
    /// SiLU activation
    Silu,
    /// Softmax
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// RMS normalization
    RMSNorm,
}

/// Tensor operations trait
pub trait TensorOps: Send + Sync {
    /// Add two tensors
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Matrix multiplication
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise multiplication
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// ReLU activation
    fn relu(&self, x: &Tensor) -> Result<Tensor>;

    /// GELU activation
    fn gelu(&self, x: &Tensor) -> Result<Tensor>;

    /// SiLU activation
    fn silu(&self, x: &Tensor) -> Result<Tensor>;

    /// Softmax
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    /// Layer normalization
    fn layer_norm(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    /// RMS normalization
    fn rms_norm(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
}

/// CPU tensor operations implementation
pub struct CpuTensorOps;

impl TensorOps for CpuTensorOps {
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        use barq_core::ops::Add;
        let op = Add;
        op.apply(a, b)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        use barq_core::ops::MatMul;
        let op = MatMul;
        op.apply(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Err(Error::Unsupported("Mul not yet implemented".to_string()))
    }

    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        use barq_core::ops::Relu;
        let op = Relu;
        op.apply(x)
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        use barq_core::ops::Gelu;
        let op = Gelu;
        op.apply(x)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        use barq_core::ops::Silu;
        let op = Silu;
        op.apply(x)
    }

    fn softmax(&self, _x: &Tensor, _dim: usize) -> Result<Tensor> {
        Err(Error::Unsupported(
            "Softmax not yet implemented".to_string(),
        ))
    }

    fn layer_norm(&self, _x: &Tensor, _dim: usize) -> Result<Tensor> {
        Err(Error::Unsupported(
            "Layer norm not yet implemented".to_string(),
        ))
    }

    fn rms_norm(&self, _x: &Tensor, _dim: usize) -> Result<Tensor> {
        Err(Error::Unsupported(
            "RMS norm not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use barq_core::tensor::{Shape, Tensor, TensorData, TensorType};

    #[test]
    fn test_tensor_ops() {
        let ops = CpuTensorOps;

        let a = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(2, 2),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();

        let result = ops.relu(&a);
        assert!(result.is_ok());
    }
}
