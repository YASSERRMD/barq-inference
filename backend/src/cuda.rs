//! CUDA backend implementation
//!
//! Provides GPU acceleration using NVIDIA CUDA with:
//! - cuBLAS for matrix operations
//! - cuDNN for neural network operations
//! - Custom kernels for quantized operations
//! - Multi-GPU support

use barq_core::error::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{safe::*, sys::*, CudaDevice, CudaStream},
    nvrtc::{safe::Ptx, CompileError},
};

/// CUDA backend
pub struct CudaBackend {
    /// CUDA device
    #[cfg(feature = "cuda")]
    pub(crate) device: Arc<CudaDevice>,
    /// Device ID
    device_id: usize,
    /// CUDA streams for concurrent execution
    #[cfg(feature = "cuda")]
    streams: HashMap<String, Arc<CudaStream>>,
    /// cuBLAS handle
    #[cfg(feature = "cuda")]
    cublas_handle: Option<cudarc::cublas::safe::Cublas>,
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
            let device = CudaDevice::new(device_id)
                .map_err(|e| Error::backend(format!("Failed to initialize CUDA device: {}", e)))?;

            // Get device properties
            let props = unsafe {
                let mut device_props: sys::CUdevice_prop = std::mem::zeroed();
                sys::cuDeviceGetProperties(&mut device_props as *mut _, device_id as i32);
                CudaDeviceProps {
                    name: {
                        let name_slice = &device_props.name;
                        let null_pos = name_slice
                            .iter()
                            .position(|&x| x == 0)
                            .unwrap_or(name_slice.len());
                        String::from_utf8_lossy(&name_slice[..null_pos]).to_string()
                    },
                    compute_capability: (device_props.major, device_props.minor),
                    total_memory: device_props.totalGlobalMem,
                    multiprocessor_count: device_props.multiProcessorCount,
                    max_threads_per_block: device_props.maxThreadsPerBlock,
                    warp_size: device_props.warpSize,
                    l2_cache_size: device_props.l2CacheSize,
                    shared_mem_per_block: device_props.sharedMemPerBlock,
                }
            };

            // Initialize cuBLAS
            let cublas_handle = cudarc::cublas::safe::Cublas::new(device.clone())
                .map(Some)
                .unwrap_or(Ok(None))?;

            Ok(Self {
                device,
                device_id,
                streams: HashMap::new(),
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
        Ok(CudaDevice::count()
            .map_err(|e| Error::backend(format!("Failed to get CUDA device count: {}", e)))?)
    }

    /// Get number of available CUDA devices
    #[cfg(not(feature = "cuda"))]
    pub fn device_count() -> Result<usize> {
        Err(Error::Unsupported("CUDA not enabled".to_string()))
    }

    /// Create or get a CUDA stream
    #[cfg(feature = "cuda")]
    pub fn get_or_create_stream(&mut self, name: &str) -> Result<Arc<CudaStream>> {
        if let Some(stream) = self.streams.get(name) {
            return Ok(stream.clone());
        }

        let stream = self
            .device
            .fork_default_stream()
            .map_err(|e| Error::backend(format!("Failed to create CUDA stream: {}", e)))?;

        self.streams.insert(name.to_string(), stream.clone());
        Ok(stream)
    }

    /// Synchronize device
    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> Result<()> {
        self.device
            .wait()
            .map_err(|e| Error::backend(format!("CUDA synchronization failed: {}", e)))
    }

    /// Get cuBLAS handle
    #[cfg(feature = "cuda")]
    pub fn cublas_handle(&self) -> Option<&cudarc::cublas::safe::Cublas> {
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
    pub buffer: cudarc::driver::safe::CudaDeviceSlice<u8>,
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
        &self,
        device: &Arc<CudaDevice>,
        data: &[T],
    ) -> Result<()> {
        let bytes = std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        );

        device
            .htod_copy_sync(&self.buffer, bytes)
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
            .dtoh_copy_sync(&self.buffer, bytes)
            .map_err(|e| Error::backend(format!("Failed to copy DtoH: {}", e)))
    }

    /// Asynchronously copy from host to device
    pub fn copy_from_host_async<T: Clone + Default>(
        &self,
        device: &Arc<CudaDevice>,
        stream: &CudaStream,
        data: &[T],
    ) -> Result<()> {
        let bytes = std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        );

        device
            .htod_copy(&self.buffer, bytes, stream)
            .map_err(|e| Error::backend(format!("Failed to async copy HtoD: {}", e)))
    }

    /// Asynchronously copy from device to host
    pub fn copy_to_host_async<T: Clone + Default>(
        &self,
        device: &Arc<CudaDevice>,
        stream: &CudaStream,
        data: &mut [T],
    ) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        device
            .dtoh_copy(&self.buffer, bytes, stream)
            .map_err(|e| Error::backend(format!("Failed to async copy DtoH: {}", e)))
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
    pub fn from_ptx(device: &Arc<CudaDevice>, ptx: &Ptx, kernel_name: &str) -> Result<Self> {
        let kernel = device
            .load_ptx(ptx, kernel_name)
            .map_err(|e| Error::backend(format!("Failed to load PTX: {}", e)))?;

        Ok(Self {
            kernel,
            device: device.clone(),
        })
    }

    /// Launch kernel
    pub fn launch(&self, cfg: LaunchConfig, params: &mut Vec<&[u8]>) -> Result<()> {
        unsafe { self.device.launch_kernel(&self.kernel, cfg, params) }
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
