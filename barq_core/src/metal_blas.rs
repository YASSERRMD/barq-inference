//! Metal-accelerated BLAS operations for Apple Silicon
//!
//! Provides GPU-accelerated matrix multiplication using Metal compute shaders.

use crate::error::{Error, Result};
use std::cell::RefCell;
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
    /// Compute pipeline for tiled matrix multiplication
    matmul_tiled_pipeline: ComputePipelineState,
    /// Compute pipeline for matrix-vector multiplication
    gemv_pipeline: ComputePipelineState,
}

#[cfg(feature = "metal")]
thread_local! {
    static METAL_SCRATCH: RefCell<Option<MetalScratch>> = RefCell::new(None);
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

        let matmul_tiled_function = library
            .get_function("matmul_tiled_kernel", None)
            .map_err(|e| Error::backend(format!("Failed to get matmul_tiled_kernel: {:?}", e)))?;
        let matmul_tiled_pipeline = device
            .new_compute_pipeline_state_with_function(&matmul_tiled_function)
            .map_err(|e| Error::backend(format!("Failed to create tiled matmul pipeline: {:?}", e)))?;

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
            matmul_tiled_pipeline,
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
    // We dispatch the grid as (columns, rows), so map x -> col and y -> row.
    uint row = gid.y;
    uint col = gid.x;

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
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE_SIZE = 16;

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

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
        METAL_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            if scratch.is_none() {
                *scratch = Some(MetalScratch::new(&self.device)?);
            }

            let scratch = scratch.as_mut().expect("scratch initialized");
            scratch.gemm(self, a, b, m, k, n)
        })
    }

    /// Matrix-vector multiplication: y = A * x
    pub fn gemv(&self, a: &[f32], x: &[f32], m: usize, n: usize) -> Result<Vec<f32>> {
        METAL_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            if scratch.is_none() {
                *scratch = Some(MetalScratch::new(&self.device)?);
            }

            let scratch = scratch.as_mut().expect("scratch initialized");
            scratch.gemv(self, a, x, m, n)
        })
    }
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct MetalScratch {
    a: MetalScratchBuffer,
    b: MetalScratchBuffer,
    c: MetalScratchBuffer,
}

#[cfg(feature = "metal")]
impl MetalScratch {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            a: MetalScratchBuffer::new(device, 0)?,
            b: MetalScratchBuffer::new(device, 0)?,
            c: MetalScratchBuffer::new(device, 0)?,
        })
    }

    fn ensure_capacity(
        &mut self,
        device: &Device,
        a_size: usize,
        b_size: usize,
        c_size: usize,
    ) -> Result<()> {
        self.a.ensure_capacity(device, a_size)?;
        self.b.ensure_capacity(device, b_size)?;
        self.c.ensure_capacity(device, c_size)?;
        Ok(())
    }

    fn gemm(
        &mut self,
        blas: &MetalBlas,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let a_len = m * k;
        let b_len = k * n;
        let c_len = m * n;
        let a_bytes = a_len * std::mem::size_of::<f32>();
        let b_bytes = b_len * std::mem::size_of::<f32>();
        let c_bytes = c_len * std::mem::size_of::<f32>();

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

        self.ensure_capacity(&blas.device, a_bytes, b_bytes, c_bytes)?;
        self.a.copy_from_f32_slice(a)?;
        self.b.copy_from_f32_slice(b)?;

        if m >= 16 && n >= 16 && k >= 16 {
            return self.gemm_tiled(blas, m, k, n, c_len);
        }

        self.gemm_naive(blas, m, k, n, c_len)
    }

    fn gemm_naive(
        &mut self,
        blas: &MetalBlas,
        m: usize,
        k: usize,
        n: usize,
        c_len: usize,
    ) -> Result<Vec<f32>> {
        let command_buffer = blas.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&blas.matmul_pipeline);
        compute_encoder.set_buffer(0, Some(&self.a.buffer), 0);
        compute_encoder.set_buffer(1, Some(&self.b.buffer), 0);
        compute_encoder.set_buffer(2, Some(&self.c.buffer), 0);

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

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut c = vec![0.0f32; c_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.c.buffer.contents() as *const f32,
                c.as_mut_ptr(),
                c_len,
            );
        }

        Ok(c)
    }

    fn gemm_tiled(
        &mut self,
        blas: &MetalBlas,
        m: usize,
        k: usize,
        n: usize,
        c_len: usize,
    ) -> Result<Vec<f32>> {
        let tile_size = 16usize;
        let tile_bytes = tile_size * tile_size * std::mem::size_of::<f32>();

        let command_buffer = blas.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&blas.matmul_tiled_pipeline);
        compute_encoder.set_buffer(0, Some(&self.a.buffer), 0);
        compute_encoder.set_buffer(1, Some(&self.b.buffer), 0);
        compute_encoder.set_buffer(2, Some(&self.c.buffer), 0);
        compute_encoder.set_threadgroup_memory_length(0, tile_bytes as u64);
        compute_encoder.set_threadgroup_memory_length(1, tile_bytes as u64);

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

        let threads_per_threadgroup = MTLSize {
            width: tile_size as u64,
            height: tile_size as u64,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: ((n + tile_size - 1) / tile_size) as u64,
            height: ((m + tile_size - 1) / tile_size) as u64,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut c = vec![0.0f32; c_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.c.buffer.contents() as *const f32,
                c.as_mut_ptr(),
                c_len,
            );
        }

        Ok(c)
    }

    fn gemv(
        &mut self,
        blas: &MetalBlas,
        a: &[f32],
        x: &[f32],
        m: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let a_len = m * n;
        let a_bytes = a_len * std::mem::size_of::<f32>();
        let x_bytes = n * std::mem::size_of::<f32>();
        let y_bytes = m * std::mem::size_of::<f32>();

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

        self.ensure_capacity(&blas.device, a_bytes, x_bytes, y_bytes)?;
        self.a.copy_from_f32_slice(a)?;
        self.b.copy_from_f32_slice(x)?;

        let command_buffer = blas.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&blas.gemv_pipeline);
        compute_encoder.set_buffer(0, Some(&self.a.buffer), 0);
        compute_encoder.set_buffer(1, Some(&self.b.buffer), 0);
        compute_encoder.set_buffer(2, Some(&self.c.buffer), 0);

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

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut y = vec![0.0f32; m];
        unsafe {
            std::ptr::copy_nonoverlapping(self.c.buffer.contents() as *const f32, y.as_mut_ptr(), m);
        }

        Ok(y)
    }
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct MetalScratchBuffer {
    buffer: Buffer,
    size: usize,
}

#[cfg(feature = "metal")]
impl MetalScratchBuffer {
    fn new(device: &Device, size: usize) -> Result<Self> {
        let size = size.max(1);
        let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        Ok(Self { buffer, size })
    }

    fn ensure_capacity(&mut self, device: &Device, size: usize) -> Result<()> {
        if size <= self.size {
            return Ok(());
        }

        *self = Self::new(device, size)?;
        Ok(())
    }

    fn copy_from_f32_slice(&self, data: &[f32]) -> Result<()> {
        let bytes = data.len() * std::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.buffer.contents() as *mut f32, data.len());
        }

        self.buffer.did_modify_range(NSRange {
            location: 0,
            length: bytes as u64,
        });
        Ok(())
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
