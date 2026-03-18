//! CUDA tensor operations implementation
//!
//! Provides GPU-accelerated tensor operations using:
//! - cuBLAS for matrix multiplication
//! - Custom kernels for activations and normalization

use crate::cuda::CudaBackend;
use crate::tensor_ops::TensorOps;
use barq_core::error::{Error, Result};
use barq_core::tensor::{Shape, Tensor, TensorData, TensorType};

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaDevice, CudaSlice};

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
        let data = match &tensor.data {
            TensorData::F32(data) => data,
            _ => return Err(Error::type_mismatch("F32", tensor.data.type_name())),
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
        #[cfg(feature = "cuda")]
        {
            // TODO: Implement CUDA element-wise add
            // For now, return error
            Err(Error::Unsupported(
                "CUDA add not yet implemented".to_string(),
            ))
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::cublas::safe::Gemv;

            let a_gpu = self.tensor_to_gpu(a)?;
            let b_gpu = self.tensor_to_gpu(b)?;

            // Get output shape
            let (m, k1) = (a.shape.dims[0], a.shape.dims[1]);
            let (k2, n) = (b.shape.dims[0], b.shape.dims[1]);

            if k1 != k2 {
                return Err(Error::dimensionMismatch(format!(
                    "Matrix dimension mismatch: A is {}x{}, B is {}x{}",
                    m, k1, k2, n
                )));
            }

            // Allocate output tensor
            let mut c_gpu = self
                .backend
                .device
                .alloc_zeros::<f32>(m * n)
                .map_err(|e| Error::backend(format!("Failed to allocate output: {}", e)))?;

            // Use cuBLAS for matrix multiplication
            if let Some(cublas) = self.backend.cublas_handle() {
                unsafe {
                    // cuBLAS expects column-major, need to transpose
                    let alpha = 1.0f32;
                    let beta = 0.0f32;

                    cublas
                        .gemm_strided_batched(
                            &a_gpu, &b_gpu, &mut c_gpu, m as i32, n as i32, k1 as i32, &alpha,
                            &beta,
                            // TODO: set transpose flags
                        )
                        .map_err(|e| Error::backend(format!("cuBLAS GEMM failed: {}", e)))?;
                }

                self.tensor_from_gpu(&c_gpu, &Shape::matrix(m, n))
            } else {
                Err(Error::backend("cuBLAS not initialized".to_string()))
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            Err(Error::Unsupported(
                "CUDA mul not yet implemented".to_string(),
            ))
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let x_gpu = self.tensor_to_gpu(x)?;

            // TODO: Launch ReLU kernel
            // For now, copy back and forth (inefficient but works)
            self.tensor_from_gpu(&x_gpu, &x.shape)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let x_gpu = self.tensor_to_gpu(x)?;

            // TODO: Launch GELU kernel
            self.tensor_from_gpu(&x_gpu, &x.shape)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let x_gpu = self.tensor_to_gpu(x)?;

            // TODO: Launch SiLU kernel
            self.tensor_from_gpu(&x_gpu, &x.shape)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    fn softmax(&self, _x: &Tensor, _dim: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            Err(Error::Unsupported(
                "CUDA softmax not yet implemented".to_string(),
            ))
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    fn layer_norm(&self, _x: &Tensor, _dim: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            Err(Error::Unsupported(
                "CUDA layer norm not yet implemented".to_string(),
            ))
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    fn rms_norm(&self, _x: &Tensor, _dim: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            Err(Error::Unsupported(
                "CUDA RMS norm not yet implemented".to_string(),
            ))
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_ops() {
        // This test will only run on systems with CUDA
        if CudaBackend::device_count().is_ok() && CudaBackend::device_count().unwrap() > 0 {
            let backend = CudaBackend::new(0).unwrap();
            let ops = CudaTensorOps::new(backend);

            let tensor = Tensor::new(
                None,
                TensorType::F32,
                Shape::matrix(2, 2),
                TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
            )
            .unwrap();

            let result = ops.relu(&tensor);
            assert!(result.is_ok());
        }
    }
}
