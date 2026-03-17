//! Buffer management for different backends

use barq_core::error::{Error, Result};
use barq_core::tensor::TensorType;

/// Buffer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferType {
    /// Read-only buffer
    ReadOnly,
    /// Write-only buffer
    WriteOnly,
    /// Read-write buffer
    ReadWrite,
}

/// Buffer trait
pub trait Buffer: Send + Sync {
    /// Returns the buffer type
    fn buffer_type(&self) -> BufferType;

    /// Returns the data type
    fn dtype(&self) -> TensorType;

    /// Returns the size in bytes
    fn size(&self) -> usize;

    /// Returns the number of elements
    fn len(&self) -> usize;

    /// Returns true if buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy data from another buffer
    fn copy_from(&mut self, src: &dyn Buffer) -> Result<()>;

    /// Clone the buffer
    fn clone_box(&self) -> Box<dyn Buffer>;

    /// Cast to Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

impl Clone for Box<dyn Buffer> {
    fn clone(&self) -> Box<dyn Buffer> {
        self.clone_box()
    }
}

/// CPU buffer implementation
pub struct CpuBuffer {
    buffer_type: BufferType,
    dtype: TensorType,
    data: Vec<u8>,
}

impl CpuBuffer {
    pub fn new(buffer_type: BufferType, dtype: TensorType, size: usize) -> Self {
        let byte_size = size * dtype.size();
        Self {
            buffer_type,
            dtype,
            data: vec![0u8; byte_size],
        }
    }

    pub fn from_vec(buffer_type: BufferType, dtype: TensorType, data: Vec<u8>) -> Result<Self> {
        if data.len() % dtype.size() != 0 {
            return Err(Error::tensor("Data size not aligned with dtype"));
        }

        Ok(Self {
            buffer_type,
            dtype,
            data,
        })
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

impl Buffer for CpuBuffer {
    fn buffer_type(&self) -> BufferType {
        self.buffer_type
    }

    fn dtype(&self) -> TensorType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn len(&self) -> usize {
        self.size() / self.dtype.size()
    }

    fn copy_from(&mut self, src: &dyn Buffer) -> Result<()> {
        if src.dtype() != self.dtype {
            return Err(Error::type_mismatch(self.dtype.name(), src.dtype().name()));
        }

        if src.size() > self.size() {
            return Err(Error::Allocation(format!(
                "Source buffer ({} bytes) larger than destination ({} bytes)",
                src.size(),
                self.size()
            )));
        }

        // Copy from CPU buffer
        if let Some(cpu_src) = src.as_any().downcast_ref::<CpuBuffer>() {
            self.data[..src.size()].copy_from_slice(&cpu_src.data[..src.size()]);
            Ok(())
        } else {
            Err(Error::Unsupported(
                "Cross-buffer copy not implemented".to_string(),
            ))
        }
    }

    fn clone_box(&self) -> Box<dyn Buffer> {
        Box::new(CpuBuffer {
            buffer_type: self.buffer_type,
            dtype: self.dtype,
            data: self.data.clone(),
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// GPU buffer placeholder
pub struct GpuBuffer {
    buffer_type: BufferType,
    dtype: TensorType,
    size: usize,
    device_id: usize,
}

impl GpuBuffer {
    pub fn new(buffer_type: BufferType, dtype: TensorType, size: usize, device_id: usize) -> Self {
        Self {
            buffer_type,
            dtype,
            size,
            device_id,
        }
    }
}

impl Buffer for GpuBuffer {
    fn buffer_type(&self) -> BufferType {
        self.buffer_type
    }

    fn dtype(&self) -> TensorType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    fn len(&self) -> usize {
        self.size / self.dtype.size()
    }

    fn copy_from(&mut self, _src: &dyn Buffer) -> Result<()> {
        Err(Error::Unsupported(
            "GPU buffer copy not implemented".to_string(),
        ))
    }

    fn clone_box(&self) -> Box<dyn Buffer> {
        Box::new(GpuBuffer {
            buffer_type: self.buffer_type,
            dtype: self.dtype,
            size: self.size,
            device_id: self.device_id,
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_buffer() {
        let buffer = CpuBuffer::new(BufferType::ReadWrite, TensorType::F32, 100);
        assert_eq!(buffer.buffer_type(), BufferType::ReadWrite);
        assert_eq!(buffer.dtype(), TensorType::F32);
        assert_eq!(buffer.len(), 100);
        assert_eq!(buffer.size(), 400); // 100 * 4 bytes
    }

    #[test]
    fn test_buffer_clone() {
        let buffer = CpuBuffer::new(BufferType::ReadWrite, TensorType::F32, 100);
        let cloned = buffer.clone_box();
        assert_eq!(cloned.len(), 100);
    }
}
