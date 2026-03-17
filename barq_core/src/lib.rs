// Barq Core - Tensor Operations and Memory Management
//
// Copyright (c) 2025 YASSERRMD <arafath.yasser@gmail.com>
//
// MIT License

// #![deny(missing_docs)]  // TODO: Re-enable after adding missing docs
#![warn(clippy::all)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod attention;
pub mod context;
pub mod error;
pub mod gemm;
pub mod gguf;
pub mod memory;
pub mod normalization;
pub mod ops;
pub mod ops_ref;
pub mod platform;
pub mod prelude;
pub mod quant;
pub mod rope;
pub mod simd;
pub mod simd_softmax;
pub mod softmax;
pub mod tensor;

pub use context::Context;
pub use error::{Error, Result};
pub use memory::{Allocator, MemoryBuffer, MemoryType};
pub use platform::{
    detect_platform, detect_simd, get_device_info, print_platform_info, DeviceInfo, PlatformType,
    SIMDCapabilities,
};
pub use quant::{Dequantize, QuantizationType, Quantize};
pub use tensor::{Tensor, TensorData, TensorType};
