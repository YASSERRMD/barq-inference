//! Prelude module for common imports

pub use crate::context::Context;
pub use crate::error::{Error, Result};
pub use crate::memory::{Allocator, MemoryBuffer, MemoryType};
pub use crate::quant::{Dequantize, QuantizationType, Quantize};
pub use crate::tensor::{Shape, Tensor, TensorType};
