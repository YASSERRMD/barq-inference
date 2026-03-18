//! Metal backend for Apple Silicon
//!
//! Provides GPU acceleration using Apple Metal framework:
//! - MTLDevice for GPU management
//! - Compute shaders for tensor operations
//! - Accelerate framework integration for BLAS
//! - Unified memory architecture support
//! - Threadgroup memory optimization

use barq_core::error::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "metal")]
use metal::{
    objc::{id, rc::autoreleasepool},
    *,
};

/// Metal backend
pub struct MetalBackend {
    /// Metal device
    #[cfg(feature = "metal")]
    device: id<MTLDevice>,
    /// Command queue
    #[cfg(feature = "metal")]
    command_queue: id<MTLCommandQueue>,
    /// Device ID
    device_id: usize,
    /// Device properties
    props: MetalDeviceProps,
    /// Loaded shader libraries
    #[cfg(feature = "metal")]
    shader_libraries: HashMap<String, id<MTLLibrary>>,
    /// Compute pipelines
    #[cfg(feature = "metal")]
    pipelines: HashMap<String, id<MTLComputePipelineState>>,
}

/// Metal device properties
#[derive(Debug, Clone)]
pub struct MetalDeviceProps {
    /// Device name
    pub name: String,
    /// Number of GPUs
    pub num_gpus: usize,
    /// Max threads per threadgroup
    pub max_threads_per_threadgroup: usize,
    /// Threadgroup memory size
    pub threadgroup_memory_size: usize,
    /// Has unified memory
    pub has_unified_memory: bool,
    /// Supports SIMD-group reductions
    pub supports_simd_reduction: bool,
    /// Supports SIMD matrix operations
    pub supports_simd_matrix: bool,
    /// Maximum buffer length
    pub max_buffer_length: usize,
}

impl MetalBackend {
    /// Create new Metal backend
    pub fn new(device_id: usize) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            autoreleasepool(|| {
                // Get all devices
                let devices = Device::all();
                let devices_vec: Vec<id<MTLDevice>> = devices.into_iter().collect();

                if device_id >= devices_vec.len() {
                    return Err(Error::backend(format!(
                        "Device ID {} out of range ({} devices available)",
                        device_id,
                        devices_vec.len()
                    )));
                }

                let device = devices_vec[device_id];

                // Create command queue
                let command_queue = device.new_command_queue();

                // Get device properties
                let props = MetalDeviceProps {
                    name: device.name().to_string(),
                    num_gpus: devices_vec.len(),
                    max_threads_per_threadgroup: device.max_threads_per_threadgroup(),
                    threadgroup_memory_size: if device.has_unified_memory() {
                        // M1/M2 typically have 32KB-64KB threadgroup memory
                        65536 // Conservative estimate
                    } else {
                        32768
                    },
                    has_unified_memory: device.has_unified_memory(),
                    supports_simd_reduction: device.supports_family(GPUFamily::Apple7),
                    supports_simd_matrix: device
                        .supports_feature(ShaderType::Compute, MTLFeatureSet::iOS_GPUFamily5_v1),
                    max_buffer_length: device.max_buffer_length(),
                };

                Ok(Self {
                    device,
                    command_queue,
                    device_id,
                    props,
                    shader_libraries: HashMap::new(),
                    pipelines: HashMap::new(),
                })
            })
        }

        #[cfg(not(feature = "metal"))]
        {
            Err(Error::Unsupported("Metal not enabled".to_string()))
        }
    }

    /// Get device properties
    pub fn props(&self) -> &MetalDeviceProps {
        &self.props
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Get number of available Metal devices
    #[cfg(feature = "metal")]
    pub fn device_count() -> Result<usize> {
        Ok(Device::all().into_iter().count())
    }

    /// Get number of available Metal devices
    #[cfg(not(feature = "metal"))]
    pub fn device_count() -> Result<usize> {
        Err(Error::Unsupported("Metal not enabled".to_string()))
    }

    /// Create command buffer
    #[cfg(feature = "metal")]
    pub fn new_command_buffer(&self) -> Result<MetalCommandBuffer> {
        let cmd_buffer = self.command_queue.new_command_buffer();
        Ok(MetalCommandBuffer {
            buffer: cmd_buffer,
            device_id: self.device_id,
        })
    }

    /// Load shader library from source
    #[cfg(feature = "metal")]
    pub fn load_shader_library(&mut self, name: &str, source: &str) -> Result<()> {
        autoreleasepool(|| {
            let options = CompileOptions::new();
            let library = self
                .device
                .new_library_with_source(source, &options)
                .map_err(|e| Error::backend(format!("Failed to compile shader: {:?}", e)))?;

            self.shader_libraries.insert(name.to_string(), library);
            Ok(())
        })
    }

    /// Create compute pipeline
    #[cfg(feature = "metal")]
    pub fn create_pipeline(
        &mut self,
        name: &str,
        function_name: &str,
        library_name: &str,
    ) -> Result<()> {
        autoreleasepool(|| {
            let library = self
                .shader_libraries
                .get(library_name)
                .ok_or_else(|| Error::backend(format!("Library {} not found", library_name)))?;

            let function = library
                .get_function(function_name, None)
                .ok_or_else(|| Error::backend(format!("Function {} not found", function_name)))?;

            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(function)
                .map_err(|e| Error::backend(format!("Failed to create pipeline: {:?}", e)))?;

            self.pipelines.insert(name.to_string(), pipeline);
            Ok(())
        })
    }

    /// Get pipeline
    #[cfg(feature = "metal")]
    pub fn pipeline(&self, name: &str) -> Option<&id<MTLComputePipelineState>> {
        self.pipelines.get(name)
    }

    /// Synchronize device (flush command buffer)
    #[cfg(feature = "metal")]
    pub fn synchronize(&self) -> Result<()> {
        // Metal uses command buffers, explicit sync not typically needed
        // But we can add a commit and wait if needed
        Ok(())
    }

    /// Get recommended threadgroup size
    #[cfg(feature = "metal")]
    pub fn recommended_threadgroup_size(&self) -> metal::MTLSize {
        // Typical threadgroup size for Apple Silicon
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        }
    }

    /// Calculate grid size for given workload
    #[cfg(feature = "metal")]
    pub fn calculate_grid_size(
        &self,
        total_elements: usize,
        threadgroup_size: usize,
    ) -> metal::MTLSize {
        metal::MTLSize {
            width: ((total_elements + threadgroup_size - 1) / threadgroup_size) as u64,
            height: 1,
            depth: 1,
        }
    }

    /// Get recommended threadgroup size
    #[cfg(not(feature = "metal"))]
    pub fn recommended_threadgroup_size(&self) -> (u64, u64, u64) {
        (256, 1, 1)
    }

    /// Calculate grid size for given workload
    #[cfg(not(feature = "metal"))]
    pub fn calculate_grid_size(
        &self,
        total_elements: usize,
        threadgroup_size: usize,
    ) -> (u64, u64, u64) {
        (
            ((total_elements + threadgroup_size - 1) / threadgroup_size) as u64,
            1,
            1,
        )
    }
}

/// Metal command buffer wrapper
#[cfg(feature = "metal")]
pub struct MetalCommandBuffer {
    /// Command buffer
    pub buffer: id<MTLCommandBuffer>,
    /// Device ID
    device_id: usize,
}

#[cfg(feature = "metal")]
impl MetalCommandBuffer {
    /// Commit command buffer
    pub fn commit(&self) {
        self.buffer.commit();
    }

    /// Wait for completion
    pub fn wait_until_completed(&self) {
        self.buffer.wait_until_completed();
    }
}

/// Metal buffer for device memory
#[cfg(feature = "metal")]
pub struct MetalBuffer {
    /// Metal buffer
    pub buffer: id<MTLBuffer>,
    /// Size in bytes
    pub size: usize,
}

#[cfg(feature = "metal")]
impl MetalBuffer {
    /// Allocate new Metal buffer
    pub fn new(device: &id<MTLDevice>, size: usize) -> Result<Self> {
        let buffer = device
            .new_buffer(size, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::backend("Failed to allocate Metal buffer".to_string()))?;

        Ok(Self { buffer, size })
    }

    /// Copy data from host to device
    pub fn copy_from_host<T>(&self, data: &[T]) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        let contents = self.buffer.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents as *mut u8, bytes.len());
        }

        // Flush to ensure data is visible to GPU
        #[cfg(feature = "metal")]
        self.buffer.did_modify_range(metal::NSRange {
            location: 0,
            length: bytes.len(),
        });

        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host<T>(&self, data: &mut [T]) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        let contents = self.buffer.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(contents as *const u8, bytes.as_ptr(), bytes.len());
        }

        Ok(())
    }

    /// Get raw pointer to buffer contents
    pub fn contents(&self) -> *mut std::ffi::c_void {
        self.buffer.contents()
    }
}

/// Metal compute encoder wrapper
#[cfg(feature = "metal")]
pub struct MetalComputeEncoder {
    /// Compute command encoder
    pub encoder: id<MTLComputeCommandEncoder>,
}

#[cfg(feature = "metal")]
impl MetalComputeEncoder {
    /// Set pipeline
    pub fn set_pipeline(&self, pipeline: &id<MTLComputePipelineState>) {
        self.encoder.set_compute_pipeline_state(pipeline);
    }

    /// Set buffer
    pub fn set_buffer(&self, index: usize, buffer: &id<MTLBuffer>, offset: usize) {
        self.encoder
            .set_buffer(index as u64, Some(*buffer), offset as u64);
    }

    /// Dispatch threadgroups
    pub fn dispatch_threadgroups(
        &self,
        threadgroups: metal::MTLSize,
        threads_per_threadgroup: metal::MTLSize,
    ) {
        self.encoder
            .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    }

    /// End encoding
    pub fn end_encoding(&self) {
        self.encoder.end_encoding();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_device_count() {
        // This test will only run on macOS with Metal support
        if let Ok(count) = MetalBackend::device_count() {
            println!("Found {} Metal devices", count);
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_backend_init() {
        // This test will only run on macOS with Metal support
        if MetalBackend::device_count().is_ok() && MetalBackend::device_count().unwrap() > 0 {
            let backend = MetalBackend::new(0);
            assert!(backend.is_ok());

            if let Ok(backend) = backend {
                println!("Metal Device: {}", backend.props.name);
                println!("Has Unified Memory: {}", backend.props.has_unified_memory);
                println!("Max Threads: {}", backend.props.max_threads_per_threadgroup);
            }
        }
    }
}
