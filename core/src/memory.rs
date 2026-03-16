//! Memory management for tensor operations

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::error::{Error, Result};

/// Memory type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// Host memory (CPU)
    Host,
    /// Device memory (GPU)
    Device,
    /// Pinned host memory
    Pinned,
    /// Unified memory (CPU + GPU)
    Unified,
}

/// Memory buffer
pub trait MemoryBuffer: Send + Sync {
    /// Returns a pointer to the data
    fn as_ptr(&self) -> *const u8;

    /// Returns a mutable pointer to the data
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// Returns the size in bytes
    fn size(&self) -> usize;

    /// Returns the memory type
    fn memory_type(&self) -> MemoryType;

    /// Copies data from another buffer
    fn copy_from(&mut self, src: &MemoryBuffer) -> Result<()>;

    /// Fills the buffer with zeros
    fn fill_zero(&mut self);
}

/// Host memory buffer
#[derive(Debug)]
pub struct HostBuffer {
    data: Vec<u8>,
    memory_type: MemoryType,
}

impl HostBuffer {
    /// Create a new host buffer
    pub fn new(size: usize) -> Result<Self> {
        Ok(Self {
            data: vec![0u8; size],
            memory_type: MemoryType::Host,
        })
    }

    /// Create a new pinned host buffer
    pub fn new_pinned(size: usize) -> Result<Self> {
        // For now, use regular allocation
        // In production, use platform-specific pinned memory allocation
        Ok(Self {
            data: vec![0u8; size],
            memory_type: MemoryType::Pinned,
        })
    }

    /// Create from existing vector
    pub fn from_vec(vec: Vec<u8>) -> Self {
        Self {
            data: vec,
            memory_type: MemoryType::Host,
        }
    }
}

impl MemoryBuffer for HostBuffer {
    fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn memory_type(&self) -> MemoryType {
        self.memory_type
    }

    fn copy_from(&mut self, src: &MemoryBuffer) -> Result<()> {
        if src.size() > self.size() {
            return Err(Error::Allocation(
                format!("Source buffer ({} bytes) larger than destination ({} bytes)",
                       src.size(), self.size())
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.as_mut_ptr(), src.size());
        }

        Ok(())
    }

    fn fill_zero(&mut self) {
        self.data.fill(0);
    }
}

/// Memory allocator trait
pub trait Allocator: Send + Sync {
    /// Allocate a memory buffer
    fn allocate(&self, size: usize) -> Result<Box<dyn MemoryBuffer>>;

    /// Returns the total memory size
    fn total_memory(&self) -> usize;

    /// Returns the used memory
    fn used_memory(&self) -> usize;

    /// Returns the available memory
    fn available_memory(&self) -> usize {
        self.total_memory().saturating_sub(self.used_memory())
    }
}

/// Default allocator using host memory
#[derive(Debug)]
pub struct DefaultAllocator {
    total_size: usize,
    used_size: Arc<AtomicUsize>,
}

impl DefaultAllocator {
    /// Create a new default allocator
    pub fn new(total_size: usize) -> Result<Self> {
        Ok(Self {
            total_size,
            used_size: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl Allocator for DefaultAllocator {
    fn allocate(&self, size: usize) -> Result<Box<dyn MemoryBuffer>> {
        let new_used = self.used_size.fetch_add(size, Ordering::Relaxed) + size;

        if new_used > self.total_size {
            self.used_size.fetch_sub(size, Ordering::Relaxed);
            return Err(Error::Allocation(
                format!("Out of memory: tried to allocate {} bytes, {} bytes already used of {} total",
                       size, new_used - size, self.total_size)
            ));
        }

        let buffer = Box::new(HostBuffer::new(size)?);
        Ok(buffer)
    }

    fn total_memory(&self) -> usize {
        self.total_size
    }

    fn used_memory(&self) -> usize {
        self.used_size.load(Ordering::Relaxed)
    }
}

impl Clone for DefaultAllocator {
    fn clone(&self) -> Self {
        Self {
            total_size: self.total_size,
            used_size: Arc::clone(&self.used_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_buffer() {
        let mut buffer = HostBuffer::new(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.memory_type(), MemoryType::Host);

        buffer.fill_zero();
        unsafe {
            assert_eq!(*buffer.as_ptr(), 0);
        }
    }

    #[test]
    fn test_allocator() {
        let allocator = DefaultAllocator::new(4096).unwrap();
        assert_eq!(allocator.total_memory(), 4096);
        assert_eq!(allocator.used_memory(), 0);

        let _buffer1 = allocator.allocate(1024).unwrap();
        assert_eq!(allocator.used_memory(), 1024);

        let _buffer2 = allocator.allocate(2048).unwrap();
        assert_eq!(allocator.used_memory(), 3072);

        // This should fail
        let result = allocator.allocate(2048);
        assert!(result.is_err());
        assert_eq!(allocator.used_memory(), 3072); // Used memory should be rolled back
    }
}
