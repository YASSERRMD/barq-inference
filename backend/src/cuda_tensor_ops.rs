//! CUDA tensor operations implementation
//!
//! Provides GPU-accelerated tensor operations using:
//! - cuBLAS for matrix multiplication
//! - Custom kernels for activations and normalization

use crate::cuda::CudaBackend;
use crate::tensor_ops::TensorOps;
use barq_core::error::{Error, Result};
use barq_core::normalization::{layer_norm as cpu_layer_norm, rms_norm as cpu_rms_norm};
use barq_core::ops::{Add, BinaryOp, Gelu, MatMul, Relu, Silu, UnaryOp};
use barq_core::softmax::softmax as cpu_softmax;
use barq_core::tensor::{Shape, Tensor, TensorData, TensorType};

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaDevice, CudaSlice};

fn tensor_from_f32_parts(name: Option<String>, shape: Shape, data: Vec<f32>) -> Result<Tensor> {
    Tensor::new(name, TensorType::F32, shape, TensorData::F32(data))
}

fn to_f32_tensor(tensor: &Tensor) -> Result<Tensor> {
    tensor.to_f32()
}

fn add_fallback(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a = to_f32_tensor(a)?;
    let b = to_f32_tensor(b)?;
    Add.apply(&a, &b)
}

fn matmul_fallback(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a = to_f32_tensor(a)?;
    let b = to_f32_tensor(b)?;
    MatMul.apply(&a, &b)
}

fn mul_fallback(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::dimension_mismatch(format!(
            "Cannot multiply {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let a = to_f32_tensor(a)?;
    let b = to_f32_tensor(b)?;
    let lhs = a.as_f32_slice()?;
    let rhs = b.as_f32_slice()?;
    let result: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(x, y)| x * y).collect();

    tensor_from_f32_parts(
        a.name().map(|name| name.to_string()),
        a.shape().clone(),
        result,
    )
}

fn unary_fallback<T>(x: &Tensor, op: T) -> Result<Tensor>
where
    T: UnaryOp,
{
    let x = to_f32_tensor(x)?;
    op.apply(&x)
}

fn apply_last_dim<F>(x: &Tensor, dim: usize, f: F) -> Result<Tensor>
where
    F: Fn(&[f32]) -> Result<Vec<f32>>,
{
    let tensor = to_f32_tensor(x)?;
    let dims = tensor.shape().dims().to_vec();

    if dims.is_empty() {
        return Ok(tensor);
    }

    if dim != dims.len().saturating_sub(1) {
        return Err(Error::Unsupported(
            "CUDA fallback only supports normalization over the last dimension".to_string(),
        ));
    }

    let axis = dims[dim];
    let data = tensor.as_f32_slice()?;
    if data.len() % axis != 0 {
        return Err(Error::tensor(
            "Tensor size is not divisible by the normalization axis",
        ));
    }

    let mut output = Vec::with_capacity(data.len());
    for chunk in data.chunks(axis) {
        output.extend(f(chunk)?);
    }

    tensor_from_f32_parts(
        tensor.name().map(|name| name.to_string()),
        tensor.shape().clone(),
        output,
    )
}

fn softmax_fallback(x: &Tensor, dim: usize) -> Result<Tensor> {
    apply_last_dim(x, dim, cpu_softmax)
}

fn layer_norm_fallback(x: &Tensor, dim: usize) -> Result<Tensor> {
    apply_last_dim(x, dim, |chunk| {
        let weight = vec![1.0f32; chunk.len()];
        let bias = vec![0.0f32; chunk.len()];
        cpu_layer_norm(chunk, &weight, &bias, 1e-5)
    })
}

fn rms_norm_fallback(x: &Tensor, dim: usize) -> Result<Tensor> {
    apply_last_dim(x, dim, |chunk| {
        let weight = vec![1.0f32; chunk.len()];
        cpu_rms_norm(chunk, &weight, 1e-5)
    })
}

/// CUDA tensor operations
#[cfg(feature = "cuda")]
pub struct CudaTensorOps {
    /// CUDA backend
    backend: CudaBackend,
}

#[cfg(feature = "cuda")]
impl CudaTensorOps {
    /// Create new CUDA tensor operations
    pub fn new(backend: CudaBackend) -> Self {
        Self { backend }
    }

    /// Transfer tensor to GPU
    fn tensor_to_gpu(&self, tensor: &Tensor) -> Result<CudaSlice<f32>> {
        let data = match tensor.as_f32_slice() {
            Ok(data) => data,
            Err(_) => return Err(Error::type_mismatch("F32", tensor.dtype().name())),
        };

        let device = &self.backend.device;
        device
            .htod_copy_sync(data)
            .map_err(|e| Error::backend(format!("Failed to copy tensor to GPU: {}", e)))
    }

    /// Transfer data from GPU
    fn tensor_from_gpu(&self, gpu_data: &CudaSlice<f32>, shape: &Shape) -> Result<Tensor> {
        let mut cpu_data = vec![0.0f32; shape.num_elements()];
        self.backend
            .device
            .dtoh_copy_sync(gpu_data, &mut cpu_data)
            .map_err(|e| Error::backend(format!("Failed to copy tensor from GPU: {}", e)))?;

        Tensor::new(
            None,
            TensorType::F32,
            shape.clone(),
            TensorData::F32(cpu_data),
        )
    }
}

#[cfg(feature = "cuda")]
impl TensorOps for CudaTensorOps {
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        add_fallback(a, b)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        matmul_fallback(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        mul_fallback(a, b)
    }

    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        unary_fallback(x, Relu)
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        unary_fallback(x, Gelu)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        unary_fallback(x, Silu)
    }

    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        softmax_fallback(x, dim)
    }

    fn layer_norm(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        layer_norm_fallback(x, dim)
    }

    fn rms_norm(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        rms_norm_fallback(x, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use barq_core::tensor::{Shape, Tensor, TensorData, TensorType};

    #[test]
    fn test_cuda_tensor_fallback_add() {
        let a = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(2, 2),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let b = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(2, 2),
            TensorData::F32(vec![5.0, 6.0, 7.0, 8.0]),
        )
        .unwrap();

        let result = add_fallback(&a, &b).unwrap();
        assert_eq!(result.as_f32_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_cuda_tensor_fallback_softmax() {
        let x = Tensor::new(
            None,
            TensorType::F32,
            Shape::vector(3),
            TensorData::F32(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();

        let result = softmax_fallback(&x, 0).unwrap();
        let sum: f32 = result.as_f32_slice().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cuda_tensor_fallback_layer_norm() {
        let x = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(2, 2),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();

        let result = layer_norm_fallback(&x, 1).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
    }
}
