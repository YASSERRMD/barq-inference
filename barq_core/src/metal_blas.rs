//! Metal-accelerated BLAS operations for Apple Silicon
//!
//! Provides GPU-accelerated matrix multiplication using Metal compute shaders.

use crate::error::{Error, Result};
use std::sync::Arc;

#[cfg(feature = "metal")]
use metal::*;

/// Metal BLAS engine for matrix operations
#[cfg(feature = "metal")]
pub struct MetalBlas {
    /// Metal device
    device: Device,
    /// Command queue
    command_queue: CommandQueue,
    /// Compute pipeline for matrix multiplication
    matmul_pipeline: ComputePipelineState,
    /// Compute pipeline for matrix-vector multiplication
    gemv_pipeline: ComputePipelineState,
}

#[cfg(feature = "metal")]
impl MetalBlas {
    /// Create new Metal BLAS engine
    pub fn new() -> Result<Self> {
        // Get default Metal device
        let device = Device::system_default()
            .ok_or_else(|| Error::backend("No Metal device available".to_string()))?;

        let command_queue = device.new_command_queue();

        // Load compute shaders
        let library = Self::load_shaders(&device)?;

        // Create matmul pipeline
        let matmul_function = library
            .get_function("matmul_kernel", None)
            .map_err(|e| Error::backend(format!("Failed to get matmul_kernel: {:?}", e)))?;
        let matmul_pipeline = device
            .new_compute_pipeline_state_with_function(&matmul_function)
            .map_err(|e| Error::backend(format!("Failed to create matmul pipeline: {:?}", e)))?;

        // Create gemv pipeline
        let gemv_function = library
            .get_function("gemv_kernel", None)
            .map_err(|e| Error::backend(format!("Failed to get gemv_kernel: {:?}", e)))?;
        let gemv_pipeline = device
            .new_compute_pipeline_state_with_function(&gemv_function)
            .map_err(|e| Error::backend(format!("Failed to create gemv pipeline: {:?}", e)))?;

        Ok(Self {
            device,
            command_queue,
            matmul_pipeline,
            gemv_pipeline,
        })
    }

    /// Load Metal compute shaders
    fn load_shaders(device: &Device) -> Result<Library> {
        let shader_code = r#"
#include <metal_stdlib>
using namespace metal;

// Matrix multiplication kernel: C = A * B
// A: (m, k), B: (k, n), C: (m, n)
kernel void matmul_kernel(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint col = gid.y;

    if (row >= m || col >= n) return;

    float sum = 0.0;
    for (uint i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + col];
    }

    C[row * n + col] = sum;
}

// Optimized matrix multiplication with tiling
kernel void matmul_tiled_kernel(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint TILE_SIZE = 32;

    uint row = gid.y * TILE_SIZE + tid.y;
    uint col = gid.x * TILE_SIZE + tid.x;

    float sum = 0.0;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load tiles into threadgroup memory
        uint a_row = row;
        uint a_col = t * TILE_SIZE + tid.x;
        uint b_row = t * TILE_SIZE + tid.y;
        uint b_col = col;

        if (a_row < M && a_col < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[a_row * K + a_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0;
        }

        if (b_row < K && b_col < N) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[b_row * N + b_col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Matrix-vector multiplication: y = A * x
kernel void gemv_kernel(
    const device float* A [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= m) return;

    float sum = 0.0;
    for (uint i = 0; i < n; i++) {
        sum += A[gid.x * n + i] * x[i];
    }

    y[gid.x] = sum;
}
"#;

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(shader_code, &options)
            .map_err(|e| Error::backend(format!("Failed to compile Metal shaders: {:?}", e)))?;

        Ok(library)
    }

    /// Matrix multiplication: C = A * B
    pub fn gemm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        let a_len = m * k;
        let b_len = k * n;
        let c_len = m * n;

        if a.len() != a_len {
            return Err(Error::tensor(format!(
                "Matrix A size mismatch: expected {}, got {}",
                a_len,
                a.len()
            )));
        }
        if b.len() != b_len {
            return Err(Error::tensor(format!(
                "Matrix B size mismatch: expected {}, got {}",
                b_len,
                b.len()
            )));
        }

        // Allocate output buffer
        let mut c = vec![0.0f32; c_len];

        // Create Metal buffers
        let buffer_a = self.device.new_buffer_with_data(
            a.as_ptr() as *const _,
            (a_len * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_b = self.device.new_buffer_with_data(
            b.as_ptr() as *const _,
            (b_len * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_c = self.device.new_buffer(
            (c_len * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();

        // Create compute encoder
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&self.matmul_pipeline);

        // Set buffers
        compute_encoder.set_buffer(0, Some(&buffer_a), 0);
        compute_encoder.set_buffer(1, Some(&buffer_b), 0);
        compute_encoder.set_buffer(2, Some(&buffer_c), 0);

        // Set parameters
        let params = [m as u32, k as u32, n as u32];
        compute_encoder.set_bytes(
            3,
            std::mem::size_of_val(&params[0]) as u64,
            &params[0] as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            4,
            std::mem::size_of_val(&params[1]) as u64,
            &params[1] as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            5,
            std::mem::size_of_val(&params[2]) as u64,
            &params[2] as *const _ as *const _,
        );

        // Dispatch threads
        let threads_per_threadgroup = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: ((n + 15) / 16) as u64,
            height: ((m + 15) / 16) as u64,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        compute_encoder.end_encoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back
        let ptr = buffer_c.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, c.as_mut_ptr(), c_len);
        }

        Ok(c)
    }

    /// Matrix-vector multiplication: y = A * x
    pub fn gemv(&self, a: &[f32], x: &[f32], m: usize, n: usize) -> Result<Vec<f32>> {
        let a_len = m * n;

        if a.len() != a_len {
            return Err(Error::tensor(format!(
                "Matrix A size mismatch: expected {}, got {}",
                a_len,
                a.len()
            )));
        }
        if x.len() != n {
            return Err(Error::tensor(format!(
                "Vector x size mismatch: expected {}, got {}",
                n,
                x.len()
            )));
        }

        // Allocate output buffer
        let mut y = vec![0.0f32; m];

        // Create Metal buffers
        let buffer_a = self.device.new_buffer_with_data(
            a.as_ptr() as *const _,
            (a_len * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_x = self.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_y = self.device.new_buffer(
            (m * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();

        // Create compute encoder
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&self.gemv_pipeline);

        // Set buffers
        compute_encoder.set_buffer(0, Some(&buffer_a), 0);
        compute_encoder.set_buffer(1, Some(&buffer_x), 0);
        compute_encoder.set_buffer(2, Some(&buffer_y), 0);

        // Set parameters
        let params = [m as u32, n as u32];
        compute_encoder.set_bytes(
            3,
            std::mem::size_of_val(&params[0]) as u64,
            &params[0] as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            4,
            std::mem::size_of_val(&params[1]) as u64,
            &params[1] as *const _ as *const _,
        );

        // Dispatch threads
        let threads_per_threadgroup = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: ((m + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        compute_encoder.end_encoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back
        let ptr = buffer_y.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, y.as_mut_ptr(), m);
        }

        Ok(y)
    }
}

#[cfg(feature = "metal")]
impl Default for MetalBlas {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal BLAS engine")
    }
}

// Stub implementation when Metal is not available
#[cfg(not(feature = "metal"))]
pub struct MetalBlas;

#[cfg(not(feature = "metal"))]
impl MetalBlas {
    pub fn new() -> Result<Self> {
        Err(Error::Unsupported("Metal not enabled".to_string()))
    }

    pub fn gemm(
        &self,
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<Vec<f32>> {
        Err(Error::Unsupported("Metal not enabled".to_string()))
    }

    pub fn gemv(&self, _a: &[f32], _x: &[f32], _m: usize, _n: usize) -> Result<Vec<f32>> {
        Err(Error::Unsupported("Metal not enabled".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_blas_creation() {
        if let Ok(blas) = MetalBlas::new() {
            println!("Metal BLAS initialized successfully");
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_gemm_simple() {
        if let Ok(blas) = MetalBlas::new() {
            let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
            let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

            let result = blas.gemm(&a, &b, 2, 2, 2).unwrap();

            // Expected: [[19, 22], [43, 50]]
            assert!((result[0] - 19.0).abs() < 1e-5);
            assert!((result[1] - 22.0).abs() < 1e-5);
            assert!((result[2] - 43.0).abs() < 1e-5);
            assert!((result[3] - 50.0).abs() < 1e-5);

            println!("Metal GEMM test passed!");
        }
    }
}
