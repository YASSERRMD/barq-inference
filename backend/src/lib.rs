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
    unreachable_code
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

//! Backend abstraction layer for Barq inference engine
//!
//! Supports multiple computation backends:
//! - CPU (with SIMD optimizations)
//! - CUDA (NVIDIA GPUs)
//! - Metal (Apple GPUs)
//! - ROCm (AMD GPUs)
//! - Vulkan (cross-platform GPU)

pub mod buffer;
pub mod compute_graph;
pub mod cpu;
pub mod cuda;
pub mod device;
pub mod metal;
pub mod tensor_ops;

pub use buffer::{Buffer, BufferType, CpuBuffer, GpuBuffer};
pub use cpu::CpuBackend;
pub use device::{CpuDevice, Device, DeviceType, GpuDevice};
pub use tensor_ops::{TensorOp, TensorOps};
