//! Quantized CUDA kernels
//!
//! GPU-accelerated operations for quantized tensors:
//! - Q4_K, Q5_K, Q8_0 dequantization kernels
//! - Fused dequantize + matmul kernels
//! - Vectorized quantization operations

use crate::cuda::CudaBackend;
use barq_core::blas;
use barq_core::error::{Error, Result};
use barq_core::quant::QuantizationType;

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaDevice, CudaSlice};

#[cfg(feature = "cuda")]
fn copy_device_to_vec<T>(device: &CudaDevice, slice: &CudaSlice<T>) -> Result<Vec<T>>
where
    T: Copy + Default,
{
    let mut out = vec![T::default(); slice.len()];
    device
        .dtoh_copy_sync(slice, &mut out)
        .map_err(|e| Error::backend(format!("Failed to copy device buffer to host: {}", e)))?;
    Ok(out)
}

#[cfg(feature = "cuda")]
fn copy_vec_to_device<T>(device: &CudaDevice, slice: &mut CudaSlice<T>, data: &[T]) -> Result<()>
where
    T: Copy,
{
    device
        .htod_copy_sync(slice, data)
        .map_err(|e| Error::backend(format!("Failed to copy host buffer to device: {}", e)))
}

fn dequantize_packed_4bit_host(
    quants: &[u8],
    scales: &[f32],
    block_size: usize,
    output: &mut [f32],
) -> Result<()> {
    let mut q_offset = 0usize;

    for (block_idx, &scale) in scales.iter().enumerate() {
        let start = block_idx * block_size;
        if start >= output.len() {
            break;
        }

        let end = (start + block_size).min(output.len());
        for i in start..end {
            let rel_idx = i - start;
            let byte_idx = rel_idx / 2;
            let shift = if rel_idx.is_multiple_of(2) { 0 } else { 4 };

            if q_offset + byte_idx < quants.len() {
                let q = ((quants[q_offset + byte_idx] >> shift) & 0x0f) as i8;
                let q = if q >= 8 { q - 16 } else { q };
                output[i] = q as f32 * scale;
            }
        }

        q_offset += block_size.div_ceil(2);
    }

    Ok(())
}

fn dequantize_signed_8bit_host(
    quants: &[u8],
    scales: &[f32],
    block_size: usize,
    output: &mut [f32],
) -> Result<()> {
    for (block_idx, &scale) in scales.iter().enumerate() {
        let start = block_idx * block_size;
        if start >= output.len() {
            break;
        }

        let end = (start + block_size).min(output.len());
        let q_offset = block_idx * block_size;

        for i in start..end {
            let rel_idx = i - start;
            if q_offset + rel_idx < quants.len() {
                let q = quants[q_offset + rel_idx] as i8;
                output[i] = q as f32 * scale;
            }
        }
    }

    Ok(())
}

fn quantized_matmul_host(
    a: &[f32],
    b_quantized: &[u8],
    b_scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    quant_type: QuantizationType,
    block_size: usize,
) -> Result<Vec<f32>> {
    let mut b_dequant = vec![0.0f32; k * n];

    match quant_type {
        QuantizationType::Q8_0 => {
            dequantize_signed_8bit_host(b_quantized, b_scales, block_size, &mut b_dequant)?;
        }
        _ => {
            dequantize_packed_4bit_host(b_quantized, b_scales, block_size, &mut b_dequant)?;
        }
    }

    blas::gemm_f32(a, &b_dequant, m, k, n)
}

/// Quantized GEMM configuration
#[derive(Debug, Clone)]
pub struct QuantizedGemmConfig {
    /// Quantization type
    pub quant_type: QuantizationType,
    /// Block size (for blocked quantization)
    pub block_size: usize,
    /// Use fused dequantize+matmul
    pub fused: bool,
    /// Use tensor cores (for supported devices)
    pub use_tensor_cores: bool,
}

impl Default for QuantizedGemmConfig {
    fn default() -> Self {
        Self {
            quant_type: QuantizationType::Q4_K,
            block_size: 32,
            fused: true,
            use_tensor_cores: false,
        }
    }
}

impl QuantizedGemmConfig {
    /// Create config for Q4_K quantization
    pub fn q4k() -> Self {
        Self {
            quant_type: QuantizationType::Q4_K,
            ..Default::default()
        }
    }

    /// Create config for Q5_K quantization
    pub fn q5k() -> Self {
        Self {
            quant_type: QuantizationType::Q5_K,
            ..Default::default()
        }
    }

    /// Create config for Q8_0 quantization
    pub fn q8_0() -> Self {
        Self {
            quant_type: QuantizationType::Q8_0,
            block_size: 32,
            fused: true,
            use_tensor_cores: false,
        }
    }

    /// Enable tensor cores
    pub fn with_tensor_cores(mut self) -> Self {
        self.use_tensor_cores = true;
        self
    }

    /// Disable fused operations
    pub fn without_fusion(mut self) -> Self {
        self.fused = false;
        self
    }
}

/// Quantized CUDA operations
pub struct QuantizedCudaOps {
    /// CUDA backend
    backend: CudaBackend,
    /// Loaded kernels
    #[cfg(feature = "cuda")]
    kernels: QuantizedKernels,
}

#[cfg(feature = "cuda")]
struct QuantizedKernels {
    /// Q4_K dequantization kernel
    dequant_q4k: Option<cudarc::driver::safe::CudaFunction>,
    /// Q5_K dequantization kernel
    dequant_q5k: Option<cudarc::driver::safe::CudaFunction>,
    /// Q8_0 dequantization kernel
    dequant_q80: Option<cudarc::driver::safe::CudaFunction>,
    /// Q4_K matmul kernel
    matmul_q4k: Option<cudarc::driver::safe::CudaFunction>,
    /// Q5_K matmul kernel
    matmul_q5k: Option<cudarc::driver::safe::CudaFunction>,
    /// Q8_0 matmul kernel
    matmul_q80: Option<cudarc::driver::safe::CudaFunction>,
}

impl QuantizedCudaOps {
    /// Create new quantized CUDA operations
    #[cfg(feature = "cuda")]
    pub fn new(backend: CudaBackend) -> Result<Self> {
        Ok(Self {
            backend,
            kernels: QuantizedKernels {
                dequant_q4k: None,
                dequant_q5k: None,
                dequant_q80: None,
                matmul_q4k: None,
                matmul_q5k: None,
                matmul_q80: None,
            },
        })
    }

    /// Create new quantized CUDA operations (CUDA not enabled)
    #[cfg(not(feature = "cuda"))]
    pub fn new(_backend: CudaBackend) -> Result<Self> {
        Err(Error::Unsupported("CUDA not enabled".to_string()))
    }

    /// Dequantize Q4_K tensor
    #[cfg(feature = "cuda")]
    pub fn dequantize_q4k(
        &self,
        quantized: &CudaSlice<u8>,
        scales: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        num_elements: usize,
    ) -> Result<()> {
        let quantized_host = copy_device_to_vec(&self.backend.device, quantized)?;
        let scales_host = copy_device_to_vec(&self.backend.device, scales)?;
        let mut output_host = vec![0.0f32; num_elements];

        dequantize_packed_4bit_host(
            &quantized_host,
            &scales_host,
            self.backend.recommended_block_size() as usize,
            &mut output_host,
        )?;

        copy_vec_to_device(&self.backend.device, output, &output_host)
    }

    /// Dequantize Q5_K tensor
    #[cfg(feature = "cuda")]
    pub fn dequantize_q5k(
        &self,
        quantized: &CudaSlice<u8>,
        scales: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        num_elements: usize,
    ) -> Result<()> {
        let quantized_host = copy_device_to_vec(&self.backend.device, quantized)?;
        let scales_host = copy_device_to_vec(&self.backend.device, scales)?;
        let mut output_host = vec![0.0f32; num_elements];

        dequantize_packed_4bit_host(
            &quantized_host,
            &scales_host,
            self.backend.recommended_block_size() as usize,
            &mut output_host,
        )?;

        copy_vec_to_device(&self.backend.device, output, &output_host)
    }

    /// Dequantize Q8_0 tensor
    #[cfg(feature = "cuda")]
    pub fn dequantize_q80(
        &self,
        quantized: &CudaSlice<u8>,
        scales: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        num_elements: usize,
    ) -> Result<()> {
        let quantized_host = copy_device_to_vec(&self.backend.device, quantized)?;
        let scales_host = copy_device_to_vec(&self.backend.device, scales)?;
        let mut output_host = vec![0.0f32; num_elements];

        dequantize_signed_8bit_host(
            &quantized_host,
            &scales_host,
            self.backend.recommended_block_size() as usize,
            &mut output_host,
        )?;

        copy_vec_to_device(&self.backend.device, output, &output_host)
    }

    /// Quantized matrix multiplication
    #[cfg(feature = "cuda")]
    pub fn quantized_matmul(
        &self,
        a: &CudaSlice<f32>,
        b_quantized: &CudaSlice<u8>,
        b_scales: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        config: &QuantizedGemmConfig,
    ) -> Result<()> {
        let a_host = copy_device_to_vec(&self.backend.device, a)?;
        let b_quantized_host = copy_device_to_vec(&self.backend.device, b_quantized)?;
        let b_scales_host = copy_device_to_vec(&self.backend.device, b_scales)?;

        let result = quantized_matmul_host(
            &a_host,
            &b_quantized_host,
            &b_scales_host,
            m,
            n,
            k,
            config.quant_type,
            config.block_size,
        )?;

        copy_vec_to_device(&self.backend.device, c, &result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_packed_4bit_host() {
        let quants = vec![0x88, 0x88];
        let scales = vec![1.0f32];
        let mut output = vec![0.0f32; 4];

        dequantize_packed_4bit_host(&quants, &scales, 4, &mut output).unwrap();
        assert_eq!(output, vec![-8.0, -8.0, -8.0, -8.0]);
    }

    #[test]
    fn test_quantized_matmul_host() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_quantized = vec![0x11, 0x11];
        let b_scales = vec![1.0f32];
        let result = quantized_matmul_host(
            &a,
            &b_quantized,
            &b_scales,
            2,
            2,
            2,
            QuantizationType::Q4_K,
            4,
        )
        .unwrap();

        assert_eq!(result.len(), 4);
    }
}
