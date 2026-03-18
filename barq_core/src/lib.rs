#![allow(
    clippy::all,
    unexpected_cfgs,
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code,
    unused_unsafe
)]
#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code,
    clippy::needless_update,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::let_and_return,
    clippy::manual_range_contains
)]
// Barq Core - Tensor Operations and Memory Management
//
// Copyright (c) 2025 YASSERRMD <arafath.yasser@gmail.com>
//
// MIT License

//   // TODO: Re-enable after adding missing docs
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod accelerate_blas;
pub mod attention;
pub mod blas;
pub mod context;
pub mod error;
pub mod gemm;
pub mod gguf;
pub mod grammar;
pub mod memory;
pub mod metal_blas;
pub mod normalization;
pub mod ops;
pub mod ops_ref;
pub mod platform;
pub mod prelude;
pub mod prompt;
pub mod quant;
pub mod rope;
pub mod simd;
pub mod simd_softmax;
pub mod softmax;
pub mod tensor;
pub mod testing;

pub use context::Context;
pub use error::{Error, Result};
pub use memory::{Allocator, MemoryBuffer, MemoryType};
pub use platform::{
    detect_platform, detect_simd, get_device_info, print_platform_info, DeviceInfo, PlatformType,
    SIMDCapabilities,
};
pub use quant::{Dequantize, QuantizationType, Quantize};
pub use tensor::{Tensor, TensorData, TensorType};

#[cfg(test)]
pub use testing::{BenchmarkTimer, TensorAssertions, TensorFixture, TestStats};
