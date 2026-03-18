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
pub mod cuda_flash_attn;
pub mod cuda_multi_gpu;
pub mod cuda_quant;
pub mod cuda_tensor_ops;
pub mod device;
pub mod metal;
pub mod tensor_ops;

pub use buffer::{Buffer, BufferType, CpuBuffer, GpuBuffer};
pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use device::{CpuDevice, Device, DeviceType, GpuDevice};
pub use tensor_ops::{TensorOp, TensorOps};

#[cfg(feature = "cuda")]
pub use cuda::{CudaBuffer, CudaDeviceProps, CudaKernel, LaunchConfig};

#[cfg(feature = "cuda")]
pub use cuda_flash_attn::{FlashAttention, FlashAttentionConfig};

#[cfg(feature = "cuda")]
pub use cuda_multi_gpu::{MultiGpuConfig, MultiGpuManager, ParallelismStrategy};

#[cfg(feature = "cuda")]
pub use cuda_quant::{QuantizedCudaOps, QuantizedGemmConfig};

#[cfg(feature = "cuda")]
pub use cuda_tensor_ops::CudaTensorOps;
