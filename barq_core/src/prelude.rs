//! Prelude module for common imports

pub use crate::error::{Error, Result};
pub use crate::tensor::{Tensor, TensorType, Shape};
pub use crate::context::Context;
pub use crate::memory::{MemoryType, MemoryBuffer, Allocator};
pub use crate::quant::{QuantizationType, Quantize, Dequantize};
