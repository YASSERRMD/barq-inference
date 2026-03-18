//! Flash Attention implementation for CUDA
//!
//! Memory-efficient attention mechanism with:
//! - O(N) memory complexity instead of O(N²)
//! - Tiled computation to fit in SRAM
//! - Online softmax algorithm
//! - Causal and bidirectional attention
//! - Support for grouped-query attention (GQA)

use crate::cuda::CudaBackend;
use barq_core::error::{Error, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaDevice, CudaSlice};

/// Flash Attention configuration
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for QK computation (default: 128)
    pub block_size_q: usize,
    /// Block size for V accumulation (default: 128)
    pub block_size_kv: usize,
    /// Use causal masking (for decoder-only models)
    pub causal: bool,
    /// Sliding window size (None = full attention)
    pub window_size: Option<usize>,
    /// Scale factor for QK dot product
    pub scale: f32,
    /// Use attention bias
    pub use_bias: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 128,
            block_size_kv: 128,
            causal: false,
            window_size: None,
            scale: 1.0 / std::f32::consts::SQRT_2,
            use_bias: false,
        }
    }
}

impl FlashAttentionConfig {
    /// Create config for causal attention (GPT-style)
    pub fn causal() -> Self {
        Self {
            causal: true,
            ..Default::default()
        }
    }

    /// Create config for sliding window attention
    pub fn sliding_window(window_size: usize) -> Self {
        Self {
            window_size: Some(window_size),
            ..Default::default()
        }
    }

    /// Set custom scale factor
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set block sizes
    pub fn with_block_sizes(mut self, q: usize, kv: usize) -> Self {
        self.block_size_q = q;
        self.block_size_kv = kv;
        self
    }

    /// Enable attention bias
    pub fn with_bias(mut self) -> Self {
        self.use_bias = true;
        self
    }
}

/// Flash Attention operator
pub struct FlashAttention {
    /// CUDA backend
    backend: CudaBackend,
    /// Loaded kernels
    #[cfg(feature = "cuda")]
    kernels: FlashAttnKernels,
}

#[cfg(feature = "cuda")]
struct FlashAttnKernels {
    /// Standard flash attention kernel
    flash_attn: Option<cudarc::driver::safe::CudaFunction>,
    /// Flash attention with causal masking
    flash_attn_causal: Option<cudarc::driver::safe::CudaFunction>,
    /// Flash attention with sliding window
    flash_attn_window: Option<cudarc::driver::safe::CudaFunction>,
    /// Flash attention with alibi bias
    flash_attn_alibi: Option<cudarc::driver::safe::CudaFunction>,
    /// Fused QKV kernel
    fused_qkv: Option<cudarc::driver::safe::CudaFunction>,
}

impl FlashAttention {
    /// Create new Flash Attention operator
    #[cfg(feature = "cuda")]
    pub fn new(backend: CudaBackend) -> Self {
        Self {
            backend,
            kernels: FlashAttnKernels {
                flash_attn: None,
                flash_attn_causal: None,
                flash_attn_window: None,
                flash_attn_alibi: None,
                fused_qkv: None,
            },
        }
    }

    /// Create new Flash Attention operator (CUDA not enabled)
    #[cfg(not(feature = "cuda"))]
    pub fn new(_backend: CudaBackend) -> Self {
        panic!("CUDA feature not enabled");
    }

    /// Compute flash attention
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, heads_k, seq_len_kv, head_dim]
    /// * `v` - Value tensor [batch, heads_k, seq_len_kv, head_dim]
    /// * `output` - Output tensor [batch, heads, seq_len, head_dim]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of query heads
    /// * `num_heads_kv` - Number of key/value heads (for GQA)
    /// * `seq_len` - Query sequence length
    /// * `seq_len_kv` - Key/Value sequence length
    /// * `head_dim` - Head dimension
    /// * `config` - Flash attention configuration
    #[cfg(feature = "cuda")]
    pub fn forward(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        num_heads_kv: usize,
        seq_len: usize,
        seq_len_kv: usize,
        head_dim: usize,
        config: &FlashAttentionConfig,
    ) -> Result<()> {
        // Validate dimensions
        if num_heads % num_heads_kv != 0 {
            return Err(Error::tensor(format!(
                "num_heads must be divisible by num_heads_kv: {} % {} != 0",
                num_heads, num_heads_kv
            )));
        }

        // Check if we can use fused kernel
        if self.can_use_fused_kernel(config) {
            self.forward_fused(
                q,
                k,
                v,
                output,
                batch_size,
                num_heads,
                num_heads_kv,
                seq_len,
                seq_len_kv,
                head_dim,
                config,
            )
        } else {
            self.forward_tiled(
                q,
                k,
                v,
                output,
                batch_size,
                num_heads,
                num_heads_kv,
                seq_len,
                seq_len_kv,
                head_dim,
                config,
            )
        }
    }

    /// Check if fused kernel can be used
    #[cfg(feature = "cuda")]
    fn can_use_fused_kernel(&self, config: &FlashAttentionConfig) -> bool {
        // Fused kernel requires specific conditions:
        // - No sliding window (or supported window size)
        // - Head dimension supported by kernel
        // - Device has sufficient shared memory

        if config.window_size.is_some() {
            return false;
        }

        // Check head dimension (typically 64, 128 supported)
        // For now, return false
        false
    }

    /// Fused flash attention kernel
    #[cfg(feature = "cuda")]
    fn forward_fused(
        &self,
        _q: &CudaSlice<f32>,
        _k: &CudaSlice<f32>,
        _v: &CudaSlice<f32>,
        _output: &mut CudaSlice<f32>,
        _batch_size: usize,
        _num_heads: usize,
        _num_heads_kv: usize,
        _seq_len: usize,
        _seq_len_kv: usize,
        _head_dim: usize,
        _config: &FlashAttentionConfig,
    ) -> Result<()> {
        // TODO: Launch fused flash attention kernel
        // This would use a single kernel for the entire attention computation
        Err(Error::Unsupported(
            "Fused flash attention kernel not yet implemented".to_string(),
        ))
    }

    /// Tiled flash attention (manual implementation)
    #[cfg(feature = "cuda")]
    fn forward_tiled(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        num_heads_kv: usize,
        seq_len: usize,
        seq_len_kv: usize,
        head_dim: usize,
        config: &FlashAttentionConfig,
    ) -> Result<()> {
        // Flash Attention algorithm:
        // 1. Partition Q, K, V into blocks
        // 2. For each Q block:
        //    a. Load Q block into SRAM
        //    b. Initialize O and l (normalization factor)
        //    c. For each K, V block:
        //       i. Load K, V blocks into SRAM
        //       ii. Compute S = Q @ K^T * scale
        //       iii. Compute P = softmax(S)
        //       iv. Update O = O + P @ V
        //       v. Update l
        //    d. Normalize O
        // 3. Write O block to HBM

        let block_q = config.block_size_q;
        let block_kv = config.block_size_kv;

        // Number of blocks
        let num_blocks_q = (seq_len + block_q - 1) / block_q;
        let num_blocks_kv = (seq_len_kv + block_kv - 1) / block_kv;

        // TODO: Launch tiled flash attention kernel
        // For now, this is a placeholder showing the algorithm structure

        // Check if causal attention requires mask
        if config.causal {
            // For causal attention, we need to mask future tokens
            // This is typically integrated into the kernel
        }

        // Check if sliding window attention
        if let Some(window) = config.window_size {
            // For sliding window, restrict attention to window_size tokens
            if window < seq_len_kv {
                // Apply window mask
            }
        }

        Err(Error::Unsupported(
            "Tiled flash attention kernel not yet implemented".to_string(),
        ))
    }

    /// Flash attention with ALiBi (Attention with Linear Biases)
    #[cfg(feature = "cuda")]
    pub fn forward_alibi(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        num_heads_kv: usize,
        seq_len: usize,
        seq_len_kv: usize,
        head_dim: usize,
        alibi_slope: f32,
    ) -> Result<()> {
        // ALiBi adds a linear bias to the attention scores
        // bias[i, j] = -slope * (j - i) for j > i

        let config = FlashAttentionConfig {
            use_bias: true,
            ..Default::default()
        };

        // TODO: Launch ALiBi flash attention kernel
        Err(Error::Unsupported(
            "ALiBi flash attention not yet implemented".to_string(),
        ))
    }

    /// Flash attention for grouped-query attention (GQA)
    #[cfg(feature = "cuda")]
    pub fn forward_gqa(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        num_heads_kv: usize,
        seq_len: usize,
        seq_len_kv: usize,
        head_dim: usize,
        config: &FlashAttentionConfig,
    ) -> Result<()> {
        // GQA uses fewer key/value heads than query heads
        // Each KV head is shared among multiple Q heads

        if num_heads_kv >= num_heads {
            return Err(Error::tensor(format!(
                "GQA requires num_heads_kv < num_heads: {} >= {}",
                num_heads_kv, num_heads
            )));
        }

        let repeats = num_heads / num_heads_kv;

        // TODO: Implement GQA flash attention
        // This requires special handling for the KV head repetition
        Err(Error::Unsupported(
            "GQA flash attention not yet implemented".to_string(),
        ))
    }

    /// Batch flash attention
    #[cfg(feature = "cuda")]
    pub fn forward_batch(
        &self,
        q_batch: &[&CudaSlice<f32>],
        k_batch: &[&CudaSlice<f32>],
        v_batch: &[&CudaSlice<f32>],
        output_batch: &mut [&mut CudaSlice<f32>],
        num_heads: usize,
        num_heads_kv: usize,
        seq_len: usize,
        seq_len_kv: usize,
        head_dim: usize,
        config: &FlashAttentionConfig,
    ) -> Result<()> {
        if q_batch.len() != k_batch.len()
            || q_batch.len() != v_batch.len()
            || q_batch.len() != output_batch.len()
        {
            return Err(Error::tensor("Batch dimensions must match"));
        }

        // Process each batch element
        for (i, ((q, k), v)) in q_batch
            .iter()
            .zip(k_batch.iter())
            .zip(v_batch.iter())
            .enumerate()
        {
            let batch_size = 1; // Each element is a single batch
            self.forward(
                q,
                k,
                v,
                output_batch[i],
                batch_size,
                num_heads,
                num_heads_kv,
                seq_len,
                seq_len_kv,
                head_dim,
                config,
            )
            .map_err(|e| Error::backend(format!("Batch element {} failed: {}", i, e)))?;
        }

        Ok(())
    }

    /// Get recommended config for device
    pub fn recommended_config(&self, seq_len: usize, causal: bool) -> FlashAttentionConfig {
        let shared_mem = self.backend.props().shared_mem_per_block;

        // Adjust block sizes based on sequence length and shared memory
        let (block_q, block_kv) = if seq_len <= 512 {
            (128, 128)
        } else if seq_len <= 2048 {
            (64, 128)
        } else {
            (32, 128)
        };

        FlashAttentionConfig {
            block_size_q: block_q,
            block_size_kv: block_kv,
            causal,
            ..Default::default()
        }
    }

    /// Check if device supports flash attention
    pub fn is_supported(&self) -> bool {
        // Flash attention requires:
        // - Sufficient shared memory
        // - Compute capability >= 7.5 (Turing) or 8.0 (Ampere) for best performance
        self.backend.props().compute_capability >= (7, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attn_config() {
        let config = FlashAttentionConfig::default();
        assert!(!config.causal);
        assert!(config.window_size.is_none());

        let config = FlashAttentionConfig::causal();
        assert!(config.causal);

        let config = FlashAttentionConfig::sliding_window(256);
        assert_eq!(config.window_size, Some(256));

        let config = FlashAttentionConfig::causal()
            .with_scale(0.5)
            .with_block_sizes(64, 128);
        assert_eq!(config.scale, 0.5);
        assert_eq!(config.block_size_q, 64);
        assert_eq!(config.block_size_kv, 128);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_flash_attention_init() {
        // This test will only run on systems with CUDA
        if CudaBackend::device_count().is_ok() && CudaBackend::device_count().unwrap() > 0 {
            let backend = CudaBackend::new(0).unwrap();
            let flash_attn = FlashAttention::new(backend);

            assert!(flash_attn.is_supported());

            let config = flash_attn.recommended_config(1024, true);
            assert!(config.causal);
            assert!(config.block_size_q > 0);
        }
    }
}
