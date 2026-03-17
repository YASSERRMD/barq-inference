//! CPU backend implementation

use std::sync::Arc;

use rayon::prelude::*;

use crate::buffer::{Buffer, CpuBuffer};
use crate::device::CpuDevice;
use barq_core::error::{Error, Result};
use barq_core::tensor::{Shape, Tensor, TensorType};

/// CPU backend
pub struct CpuBackend {
    device: Arc<CpuDevice>,
    num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            device: Arc::new(CpuDevice::new()),
            num_threads: rayon::current_num_threads(),
        }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        let device = Arc::new(CpuDevice::with_threads(num_threads));
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok();

        Self {
            device,
            num_threads,
        }
    }

    pub fn device(&self) -> &CpuDevice {
        &self.device
    }

    /// Allocate a buffer on the CPU
    pub fn allocate(&self, dtype: TensorType, size: usize) -> Result<CpuBuffer> {
        Ok(CpuBuffer::new(
            crate::buffer::BufferType::ReadWrite,
            dtype,
            size,
        ))
    }

    /// Copy data to CPU buffer
    pub fn copy_to_buffer(&self, data: &[u8], buffer: &mut CpuBuffer) -> Result<()> {
        if data.len() > buffer.size() {
            return Err(Error::Allocation(format!(
                "Data size ({} bytes) exceeds buffer size ({} bytes)",
                data.len(),
                buffer.size()
            )));
        }

        let buffer_data = buffer.data_mut();
        buffer_data[..data.len()].copy_from_slice(data);

        Ok(())
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let backend = CpuBackend::new();
        assert_eq!(backend.num_threads, rayon::current_num_threads());

        let buffer = backend.allocate(TensorType::F32, 100).unwrap();
        assert_eq!(buffer.len(), 100);
    }

    #[test]
    fn test_cpu_backend_with_threads() {
        let backend = CpuBackend::with_threads(4);
        assert_eq!(backend.num_threads, 4);
    }
}
