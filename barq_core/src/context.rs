//! Computation context for tensor operations
//!
//! The context manages memory allocation and computation graphs.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};
use crate::memory::{Allocator, MemoryBuffer};
use crate::tensor::{Shape, Tensor, TensorType};

/// Computation context
#[derive(Clone)]
pub struct Context {
    /// Context ID
    id: usize,
    /// Memory allocator
    allocator: Arc<dyn Allocator>,
    /// Named tensors
    tensors: Arc<RwLock<HashMap<String, Tensor>>>,
    /// Context parameters
    params: ContextParams,
}

/// Context parameters
#[derive(Debug, Clone)]
pub struct ContextParams {
    /// Memory buffer size in bytes
    pub mem_size: usize,
    /// Maximum number of tensors
    pub max_tensors: usize,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            mem_size: 16 * 1024 * 1024, // 16 MB
            max_tensors: 1024,
        }
    }
}

impl Context {
    /// Create a new context
    pub fn new(params: ContextParams) -> Result<Self> {
        use crate::memory::DefaultAllocator;

        let allocator = Arc::new(DefaultAllocator::new(params.mem_size)?);
        let id = allocate_context_id();

        Ok(Self {
            id,
            allocator,
            tensors: Arc::new(RwLock::new(HashMap::new())),
            params,
        })
    }

    /// Create a new context with default parameters
    pub fn default() -> Result<Self> {
        Self::new(ContextParams::default())
    }

    /// Returns the context ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Allocate a new tensor
    pub fn allocate_tensor(&self, name: String, dtype: TensorType, shape: Shape) -> Result<Tensor> {
        let size = shape.num_elements() * dtype.size();

        let buffer = self.allocator.allocate(size)?;

        let data = match dtype {
            TensorType::F32 => {
                let vec = vec![0.0f32; shape.num_elements()];
                crate::tensor::TensorData::F32(vec)
            }
            _ => {
                return Err(Error::Unsupported(format!(
                    "Allocate tensor not implemented for {}",
                    dtype
                )))
            }
        };

        let tensor = Tensor::new(Some(name.clone()), dtype, shape, data)?;

        let mut tensors = self
            .tensors
            .write()
            .map_err(|e| Error::Backend(format!("Failed to acquire write lock: {}", e)))?;

        tensors.insert(name, tensor.clone());

        Ok(tensor)
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<Tensor> {
        let tensors = self.tensors.read().ok()?;
        tensors.get(name).cloned()
    }

    /// Remove a tensor by name
    pub fn remove_tensor(&self, name: &str) -> Result<Tensor> {
        let mut tensors = self
            .tensors
            .write()
            .map_err(|e| Error::Backend(format!("Failed to acquire write lock: {}", e)))?;

        tensors
            .remove(name)
            .ok_or_else(|| Error::tensor(format!("Tensor '{}' not found", name)))
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        let tensors = self.tensors.read().ok();
        tensors
            .map(|t| t.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns the total used memory in bytes
    pub fn used_memory(&self) -> usize {
        self.allocator.used_memory()
    }

    /// Returns the total memory size in bytes
    pub fn total_memory(&self) -> usize {
        self.allocator.total_memory()
    }
}

/// Atomic counter for context IDs
static NEXT_CONTEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

fn allocate_context_id() -> usize {
    NEXT_CONTEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = Context::default().unwrap();
        assert!(ctx.id() > 0);
        assert!(ctx.total_memory() > 0);
    }

    #[test]
    fn test_tensor_allocation() {
        let ctx = Context::default().unwrap();

        let tensor = ctx
            .allocate_tensor("test".to_string(), TensorType::F32, Shape::matrix(2, 3))
            .unwrap();

        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(tensor.dtype(), TensorType::F32);
    }

    #[test]
    fn test_tensor_retrieval() {
        let ctx = Context::default().unwrap();

        ctx.allocate_tensor("test".to_string(), TensorType::F32, Shape::vector(10))
            .unwrap();

        let tensor = ctx.get_tensor("test");
        assert!(tensor.is_some());
        assert_eq!(tensor.unwrap().num_elements(), 10);
    }
}
