//! Backend abstraction layer for Barq inference engine
//!
//! Supports multiple computation backends:
//! - CPU (with SIMD optimizations)
//! - CUDA (NVIDIA GPUs)
//! - Metal (Apple GPUs)
//! - ROCm (AMD GPUs)
//! - Vulkan (cross-platform GPU)

pub mod device;
pub mod buffer;
pub mod cpu;
pub mod cuda;
pub mod metal;
pub mod tensor_ops;
pub mod compute_graph;

pub use device::{Device, DeviceType, CpuDevice, GpuDevice};
pub use buffer::{Buffer, BufferType, CpuBuffer, GpuBuffer};
pub use cpu::CpuBackend;
pub use tensor_ops::{TensorOps, TensorOp};
