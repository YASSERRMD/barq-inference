//! Metal backend placeholder

use core::tensor::{Tensor, TensorType, Shape};
use core::error::{Error, Result};

/// Metal backend (placeholder for future implementation)
pub struct MetalBackend {
    device_id: usize,
}

impl MetalBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            // TODO: Initialize Metal
            Ok(Self { device_id })
        }

        #[cfg(not(feature = "metal"))]
        {
            Err(Error::Unsupported("Metal not enabled".to_string()))
        }
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}
