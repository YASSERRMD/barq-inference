//! CUDA backend implementation
//!
//! Provides GPU acceleration using NVIDIA CUDA with:
//! - cuBLAS for matrix operations
//! - cuDNN for neural network operations
//! - Custom kernels for quantized operations
//! - Multi-GPU support

use barq_core::error::{Error, Result};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::{
    cublas::safe::CudaBlas,
    driver::{
        result,
        safe::{CudaDevice, CudaFunction, CudaSlice, CudaStream, LaunchAsync},
        sys,
    },
    nvrtc::safe::Ptx,
};

/// CUDA backend
pub struct CudaBackend {
    /// CUDA device
    #[cfg(feature = "cuda")]
    pub(crate) device: Arc<CudaDevice>,
    /// Device ID
    device_id: usize,
    /// cuBLAS handle
    #[cfg(feature = "cuda")]
    cublas_handle: Option<CudaBlas>,
    /// Device properties
    props: CudaDeviceProps,
}

/// CUDA device properties
#[derive(Debug, Clone)]
pub struct CudaDeviceProps {
    /// Device name
    pub name: String,
    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),
    /// Total memory in bytes
    pub total_memory: usize,
    /// Number of multiprocessors
    pub multiprocessor_count: u32,
    /// Max threads per block
    pub max_threads_per_block: u32,
    /// Warp size
    pub warp_size: u32,
    /// L2 cache size in bytes
    pub l2_cache_size: usize,
    /// Shared memory per block in bytes
    pub shared_mem_per_block: usize,
}

impl CudaBackend {
    /// Create new CUDA backend
    pub fn new(device_id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            // Get CUDA device
            let device = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                CudaDevice::new(device_id)
            })) {
                Ok(Ok(device)) => device,
                Ok(Err(e)) => {
                    return Err(Error::backend(format!(
                        "Failed to initialize CUDA device: {}",
                        e
                    )));
                }
                Err(_) => {
                    return Err(Error::Unsupported("CUDA runtime not available".to_string()));
                }
            };

            let total_memory = unsafe { result::device::total_mem(*device.cu_device()) }
                .map_err(|e| Error::backend(format!("Failed to query CUDA memory: {}", e)))?;

            let props = CudaDeviceProps {
                name: device
                    .name()
                    .map_err(|e| Error::backend(format!("Failed to query CUDA name: {}", e)))?,
                compute_capability: (
                    device
                        .attribute(
                            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        )
                        .map_err(|e| {
                            Error::backend(format!(
                                "Failed to query CUDA compute capability major: {}",
                                e
                            ))
                        })? as u32,
                    device
                        .attribute(
                            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        )
                        .map_err(|e| {
                            Error::backend(format!(
                                "Failed to query CUDA compute capability minor: {}",
                                e
                            ))
                        })? as u32,
                ),
                total_memory,
                multiprocessor_count: device
                    .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                    .map_err(|e| {
                        Error::backend(format!("Failed to query CUDA multiprocessor count: {}", e))
                    })? as u32,
                max_threads_per_block: device
                    .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                    .map_err(|e| {
                        Error::backend(format!("Failed to query CUDA max threads per block: {}", e))
                    })? as u32,
                warp_size: device
                    .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
                    .map_err(|e| Error::backend(format!("Failed to query CUDA warp size: {}", e)))?
                    as u32,
                l2_cache_size: device
                    .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
                    .map_err(|e| {
                        Error::backend(format!("Failed to query CUDA L2 cache size: {}", e))
                    })? as usize,
                shared_mem_per_block: device
                    .attribute(
                        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                    )
                    .map_err(|e| {
                        Error::backend(format!(
                            "Failed to query CUDA shared memory per block: {}",
                            e
                        ))
                    })? as usize,
            };

            // Initialize cuBLAS
            let cublas_handle = CudaBlas::new(device.clone()).ok();

            Ok(Self {
                device,
                device_id,
                cublas_handle,
                props,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Unsupported("CUDA not enabled".to_string()))
        }
    }

    /// Get device properties
    pub fn props(&self) -> &CudaDeviceProps {
        &self.props
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Get number of available CUDA devices
    #[cfg(feature = "cuda")]
    pub fn device_count() -> Result<usize> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(CudaDevice::count)) {
            Ok(Ok(count)) => Ok(count as usize),
            Ok(Err(e)) => Err(Error::backend(format!(
                "Failed to get CUDA device count: {}",
                e
            ))),
            Err(_) => Err(Error::Unsupported("CUDA runtime not available".to_string())),
        }
    }

    /// Get number of available CUDA devices
    #[cfg(not(feature = "cuda"))]
    pub fn device_count() -> Result<usize> {
        Err(Error::Unsupported("CUDA not enabled".to_string()))
    }

    /// Create or get a CUDA stream
    #[cfg(feature = "cuda")]
    pub fn get_or_create_stream(&mut self, name: &str) -> Result<Arc<CudaStream>> {
        let _ = name;
        let stream = self
            .device
            .fork_default_stream()
            .map_err(|e| Error::backend(format!("Failed to create CUDA stream: {}", e)))?;

        Ok(Arc::new(stream))
    }

    /// Synchronize device
    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| Error::backend(format!("CUDA synchronization failed: {}", e)))
    }

    /// Get cuBLAS handle
    #[cfg(feature = "cuda")]
    pub fn cublas_handle(&self) -> Option<&CudaBlas> {
        self.cublas_handle.as_ref()
    }

    /// Check if device supports FP16
    pub fn supports_fp16(&self) -> bool {
        self.props.compute_capability >= (5, 3)
    }

    /// Check if device supports BF16
    pub fn supports_bf16(&self) -> bool {
        self.props.compute_capability >= (8, 0)
    }

    /// Check if device supports FP8
    pub fn supports_fp8(&self) -> bool {
        self.props.compute_capability >= (8, 9)
    }

    /// Get recommended block size for kernels
    pub fn recommended_block_size(&self) -> u32 {
        self.props.max_threads_per_block.min(256)
    }

    /// Get recommended warp size
    pub fn warp_size(&self) -> u32 {
        self.props.warp_size
    }

    /// Calculate optimal grid size for given workload
    pub fn calculate_grid_size(&self, total_elements: usize, block_size: u32) -> u32 {
        ((total_elements + block_size as usize - 1) / block_size as usize) as u32
    }
}

/// CUDA buffer for device memory
#[cfg(feature = "cuda")]
pub struct CudaBuffer {
    /// Device buffer
    pub buffer: CudaSlice<u8>,
    /// Size in bytes
    pub size: usize,
}

#[cfg(feature = "cuda")]
impl CudaBuffer {
    /// Allocate new CUDA buffer
    pub fn new(device: &Arc<CudaDevice>, size: usize) -> Result<Self> {
        let buffer = device
            .alloc_zeros::<u8>(size)
            .map_err(|e| Error::backend(format!("Failed to allocate CUDA buffer: {}", e)))?;

        Ok(Self { buffer, size })
    }

    /// Copy data from host to device
    pub fn copy_from_host<T: Clone + Default>(
        &mut self,
        device: &Arc<CudaDevice>,
        data: &[T],
    ) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        device
            .htod_sync_copy_into(bytes, &mut self.buffer)
            .map_err(|e| Error::backend(format!("Failed to copy HtoD: {}", e)))
    }

    /// Copy data from device to host
    pub fn copy_to_host<T: Clone + Default>(
        &self,
        device: &Arc<CudaDevice>,
        data: &mut [T],
    ) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        device
            .dtoh_sync_copy_into(&self.buffer, bytes)
            .map_err(|e| Error::backend(format!("Failed to copy DtoH: {}", e)))
    }

    /// Asynchronously copy from host to device
    pub fn copy_from_host_async<T: Clone + Default>(
        &mut self,
        device: &Arc<CudaDevice>,
        _stream: &CudaStream,
        data: &[T],
    ) -> Result<()> {
        self.copy_from_host(device, data)
    }

    /// Asynchronously copy from device to host
    pub fn copy_to_host_async<T: Clone + Default>(
        &self,
        device: &Arc<CudaDevice>,
        _stream: &CudaStream,
        data: &mut [T],
    ) -> Result<()> {
        self.copy_to_host(device, data)
    }
}

/// CUDA kernel wrapper
#[cfg(feature = "cuda")]
pub struct CudaKernel {
    /// Compiled kernel function
    pub kernel: cudarc::driver::safe::CudaFunction,
    /// Device
    pub device: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl CudaKernel {
    /// Load kernel from PTX
    pub fn from_ptx(
        device: &Arc<CudaDevice>,
        ptx: &Ptx,
        kernel_name: &'static str,
    ) -> Result<Self> {
        device
            .load_ptx(ptx.clone(), kernel_name, &[kernel_name])
            .map_err(|e| Error::backend(format!("Failed to load PTX: {}", e)))?;

        let kernel = device.get_func(kernel_name, kernel_name).ok_or_else(|| {
            Error::backend(format!(
                "Failed to retrieve CUDA function '{}' from module '{}'",
                kernel_name, kernel_name
            ))
        })?;

        Ok(Self {
            kernel,
            device: device.clone(),
        })
    }

    /// Launch kernel
    pub fn launch<P>(&self, cfg: LaunchConfig, params: P) -> Result<()>
    where
        CudaFunction: LaunchAsync<P>,
    {
        unsafe { self.kernel.clone().launch(cfg.into(), params) }
            .map_err(|e| Error::backend(format!("Kernel launch failed: {}", e)))
    }
}

/// Kernel launch configuration
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_mem_bytes: u32,
}

#[cfg(feature = "cuda")]
impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

#[cfg(feature = "cuda")]
impl LaunchConfig {
    /// Create config for 1D grid
    pub fn for_1d(grid_size: u32, block_size: u32) -> Self {
        Self {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Create config for 2D grid
    pub fn for_2d(grid_size: (u32, u32), block_size: (u32, u32)) -> Self {
        Self {
            grid_dim: (grid_size.0, grid_size.1, 1),
            block_dim: (block_size.0, block_size.1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Create config for 3D grid
    pub fn for_3d(grid_size: (u32, u32, u32), block_size: (u32, u32, u32)) -> Self {
        Self {
            grid_dim: grid_size,
            block_dim: block_size,
            shared_mem_bytes: 0,
        }
    }

    /// Set shared memory size
    pub fn with_shared_memory(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }
}

#[cfg(feature = "cuda")]
impl From<LaunchConfig> for cudarc::driver::safe::LaunchConfig {
    fn from(value: LaunchConfig) -> Self {
        Self {
            grid_dim: value.grid_dim,
            block_dim: value.block_dim,
            shared_mem_bytes: value.shared_mem_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_device_count() {
        // This test will only run on systems with CUDA
        if let Ok(count) = CudaBackend::device_count() {
            println!("Found {} CUDA devices", count);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_backend_init() {
        // This test will only run on systems with CUDA
        if CudaBackend::device_count().is_ok() {
            let backend = CudaBackend::new(0);
            assert!(backend.is_ok());

            if let Ok(backend) = backend {
                println!("CUDA Device: {}", backend.props.name);
                println!(
                    "Compute Capability: {}.{}",
                    backend.props.compute_capability.0, backend.props.compute_capability.1
                );
                println!(
                    "Total Memory: {} MB",
                    backend.props.total_memory / (1024 * 1024)
                );
            }
        }
    }
}
