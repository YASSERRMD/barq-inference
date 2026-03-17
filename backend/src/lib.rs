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
