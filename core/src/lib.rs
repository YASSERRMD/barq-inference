// Barq Core - Tensor Operations and Memory Management
//
// Copyright (c) 2025 YASSERRMD <arafath.yasser@gmail.com>
//
// MIT License

#![deny(missing_docs)]
#![warn(clippy::all)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod error;
pub mod tensor;
pub mod context;
pub mod memory;
pub mod ops;
pub mod ops_ref;
pub mod gguf;
pub mod quant;
pub mod softmax;
pub mod normalization;
pub mod attention;
pub mod rope;
pub mod simd;
pub mod simd_softmax;
pub mod gemm;
pub mod platform;
pub mod prelude;

pub use error::{Error, Result};
pub use tensor::{Tensor, TensorType, TensorData};
pub use context::Context;
pub use memory::{MemoryType, MemoryBuffer, Allocator};
pub use quant::{QuantizationType, Quantize, Dequantize};
pub use platform::{PlatformType, SIMDCapabilities, DeviceInfo, detect_platform, detect_simd, get_device_info, print_platform_info};
