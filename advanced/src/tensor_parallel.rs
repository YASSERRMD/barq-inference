//! Tensor parallelism for distributed inference
//!
//! Splits model tensors across multiple devices/GPUs to:
//! - Run models larger than single device memory
//! - Distribute computation for faster inference

use std::sync::Arc;

use tokio::sync::RwLock;

use barq_core::error::{Error, Result};
use barq_core::tensor::{Shape, Tensor};

/// Device ID type
pub type DeviceId = usize;

/// Tensor parallel configuration
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Number of devices
    pub n_devices: usize,
    /// Communication backend
    pub backend: CommBackend,
}

/// Communication backend for tensor parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommBackend {
    /// NCCL (for NVIDIA GPUs)
    Nccl,
    /// MPI
    Mpi,
    /// Gloo
    Gloo,
    /// Custom allreduce
    Custom,
}

/// Tensor parallel runtime
pub struct TensorParallel {
    config: TensorParallelConfig,
    /// Device ID this rank is responsible for
    device_id: DeviceId,
    /// Local tensor shards
    local_shards: Arc<RwLock<Vec<Option<Tensor>>>>,
}

impl TensorParallel {
    /// Create a new tensor parallel runtime
    pub fn new(config: TensorParallelConfig, device_id: DeviceId) -> Result<Self> {
        if device_id >= config.n_devices {
            return Err(Error::Backend(format!(
                "Invalid device_id {} >= n_devices {}",
                device_id, config.n_devices
            )));
        }

        let local_shards = Arc::new(RwLock::new(vec![None; config.n_devices]));

        Ok(Self {
            config,
            device_id,
            local_shards,
        })
    }

    /// Split a tensor across devices
    pub fn split_tensor(&self, tensor: &Tensor, dim: usize) -> Result<Vec<Tensor>> {
        let shape = tensor.shape();
        let dims = shape.dims();

        if dim >= dims.len() {
            return Err(Error::tensor(format!(
                "Cannot split along dim {}, tensor has {} dims",
                dim,
                dims.len()
            )));
        }

        let n_devices = self.config.n_devices;
        let dim_size = dims[dim];

        if !dim_size.is_multiple_of(n_devices) {
            return Err(Error::tensor(format!(
                "Dimension {} (size {}) not divisible by n_devices {}",
                dim, dim_size, n_devices
            )));
        }

        let chunk_size = dim_size / n_devices;
        let mut shards = Vec::with_capacity(n_devices);

        // TODO: Implement actual tensor splitting
        // For now, return dummy shards
        for i in 0..n_devices {
            let mut new_dims = dims.to_vec();
            new_dims[dim] = chunk_size;

            let new_shape = Shape::new(new_dims);
            let shard = Tensor::zeros(tensor.dtype(), new_shape);
            shards.push(shard);
        }

        Ok(shards)
    }

    /// All-reduce operation across devices
    pub async fn all_reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        // TODO: Implement actual all-reduce
        // For single-device, just return the tensor
        if self.config.n_devices == 1 {
            return Ok(tensor.clone());
        }

        Err(Error::Unsupported(
            "All-reduce not yet implemented".to_string(),
        ))
    }

    /// Broadcast a tensor from device 0 to all devices
    pub async fn broadcast(&self, tensor: &Tensor) -> Result<()> {
        // TODO: Implement actual broadcast
        Ok(())
    }

    /// Returns the device ID
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Returns the number of devices
    pub fn n_devices(&self) -> usize {
        self.config.n_devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_parallel() {
        let config = TensorParallelConfig {
            n_devices: 4,
            backend: CommBackend::Custom,
        };

        let tp = TensorParallel::new(config, 0);
        assert!(tp.is_ok());

        let tp = tp.unwrap();
        assert_eq!(tp.device_id(), 0);
        assert_eq!(tp.n_devices(), 4);
    }

    #[test]
    fn test_invalid_device_id() {
        let config = TensorParallelConfig {
            n_devices: 2,
            backend: CommBackend::Custom,
        };

        let tp = TensorParallel::new(config, 5);
        assert!(tp.is_err());
    }
}
