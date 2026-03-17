//! Flash Attention-2 implementation with tiling
//!
//! Implements the Flash Attention algorithm from "Flash Attention: Fast and Memory-Efficient
//! Exact Attention with IO-Awareness" which reduces memory complexity from O(N²) to O(N)
//! and improves cache efficiency through tiling.
//!
//! Key optimizations:
//! - Online softmax: computes softmax incrementally to avoid materializing the full N×N matrix
//! - Tiling: processes attention in blocks to fit in CPU cache (L2/L3)
//! - Recomputation: avoids storing the full attention matrix
//! - SIMD-friendly operations for better vectorization

use barq_core::error::{Error, Result};
use barq_core::tensor::{Tensor, TensorType, TensorData, Shape};

/// Flash Attention-2 configuration
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for QK dot product (default: 128)
    pub block_size_qk: usize,
    /// Block size for softmax (default: 128)
    pub block_size_soft: usize,
    /// Whether to use causal masking (for autoregressive models)
    pub causal: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_qk: 128,
            block_size_soft: 128,
            causal: true,
        }
    }
}

/// Flash Attention-2 implementation
pub struct FlashAttention {
    config: FlashAttentionConfig,
    num_heads: usize,
    head_dim: usize,
}

impl FlashAttention {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            config: FlashAttentionConfig::default(),
            num_heads,
            head_dim,
        }
    }

    pub fn with_config(mut self, config: FlashAttentionConfig) -> Self {
        self.config = config;
        self
    }

    /// Forward pass with Flash Attention-2
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq_len, num_heads, head_dim]
    /// * `k` - Key tensor [batch, seq_len, num_heads, head_dim]
    /// * `v` - Value tensor [batch, seq_len, num_heads, head_dim]
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, num_heads, head_dim]
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        // Validate inputs
        self.validate_inputs(q, k, v)?;

        let shape = q.shape();
        let batch_size = shape\.dims()[0];
        let seq_len = shape\.dims()[1];

        // Extract f32 data
        let q_data = q.as_f32_slice()?;
        let k_data = k.as_f32_slice()?;
        let v_data = v.as_f32_slice()?;

        // Output buffer
        let output_size = batch_size * seq_len * self.num_heads * self.head_dim;
        let mut output = vec![0.0f32; output_size];

        // Process each batch
        for b in 0..batch_size {
            self.forward_batch(
                b,
                q_data,
                k_data,
                v_data,
                seq_len,
                &mut output,
            )?;
        }

        Ok(Tensor::new(
            None,
            TensorType::F32,
            shape.clone(),
            TensorData::F32(output),
        )?)
    }

    /// Compute attention with causal masking (alias for clarity)
    pub fn forward_causal(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let mut config = self.config.clone();
        config.causal = true;
        let fa = FlashAttention {
            config,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
        };
        fa.forward(q, k, v)
    }

    fn validate_inputs(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<()> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();

        if q_shape.dims.len() != 4 || k_shape.dims.len() != 4 || v_shape.dims.len() != 4 {
            return Err(Error::tensor(
                "Q, K, V must be 4D tensors [batch, seq, heads, head_dim]",
            ));
        }

        if q_shape != k_shape || q_shape != v_shape {
            return Err(Error::tensor("Q, K, V must have the same shape"));
        }

        if q_shape\.dims()[2] != self.num_heads {
            return Err(Error::tensor("Number of heads mismatch"));
        }

        if q_shape\.dims()[3] != self.head_dim {
            return Err(Error::tensor("Head dimension mismatch"));
        }

        Ok(())
    }

    fn forward_batch(
        &self,
        batch_idx: usize,
        q_data: &[f32],
        k_data: &[f32],
        v_data: &[f32],
        seq_len: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let batch_stride = seq_len * self.num_heads * self.head_dim;

        // Process each attention head
        for head in 0..self.num_heads {
            let head_stride = seq_len * self.head_dim;

            // Extract Q, K, V for this head
            let q_head = &q_data[batch_idx * batch_stride + head * head_stride..][..seq_len * self.head_dim];
            let k_head = &k_data[batch_idx * batch_stride + head * head_stride..][..seq_len * self.head_dim];
            let v_head = &v_data[batch_idx * batch_stride + head * head_stride..][..seq_len * self.head_dim];

            // Compute Flash Attention for this head
            let out_start = batch_idx * batch_stride + head * head_stride;
            let out_slice = &mut output[out_start..out_start + seq_len * self.head_dim];

            self.flash_attention_tiled(q_head, k_head, v_head, seq_len, out_slice)?;
        }

        Ok(())
    }

    /// Core Flash Attention algorithm with tiling
    fn flash_attention_tiled(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let block_size = self.config.block_size_qk;

        // Initialize output and running statistics
        // O[i] = 0 for all i
        // l[i] = 0 (logsumexp)
        // m[i] = -inf (max)
        let mut o = vec![0.0f32; seq_len * self.head_dim];
        let mut l = vec![0.0f32; seq_len];
        let mut m = vec![f32::NEG_INFINITY; seq_len];

        // Process Key/Value in blocks (tr blocks)
        let num_tr_blocks = (seq_len + block_size - 1) / block_size;

        for tr_block_idx in 0..num_tr_blocks {
            let tr_start = tr_block_idx * block_size;
            let tr_end = (tr_start + block_size).min(seq_len);

            // Compute QK^T for this block: Q[i] @ K[tr_start:tr_end]^T
            // Shape: [seq_len, block_size]
            let mut qk_block = vec![0.0f32; seq_len * (tr_end - tr_start)];

            for i in 0..seq_len {
                for j in tr_start..tr_end {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q[i * self.head_dim + d] * k[j * self.head_dim + d];
                    }
                    // Apply scaling
                    qk_block[i * (tr_end - tr_start) + (j - tr_start)] = dot / (self.head_dim as f32).sqrt();
                }
            }

            // Apply causal mask if needed
            if self.config.causal {
                for i in 0..seq_len {
                    for j in tr_start..tr_end {
                        if j > i {
                            qk_block[i * (tr_end - tr_start) + (j - tr_start)] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Online softmax update for each query position
            for i in 0..seq_len {
                // Find max in this block for position i
                let block_max = qk_block[i * (tr_end - tr_start)..(i + 1) * (tr_end - tr_start)]
                    .iter()
                    .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

                // Update global max
                let new_m = m[i].max(block_max);

                // Compute exp and sum
                let mut block_exp_sum = 0.0f32;
                for j in tr_start..tr_end {
                    let attn_val = qk_block[i * (tr_end - tr_start) + (j - tr_start)];
                    if attn_val != f32::NEG_INFINITY {
                        block_exp_sum += (attn_val - new_m).exp();
                    }
                }

                // Update statistics
                let alpha = (m[i] - new_m).exp();
                let new_l = l[i] * alpha + block_exp_sum;

                // Update output: O[i] = alpha * O[i] + softmax(QK[i, tr_block]) @ V[tr_block]
                // Scale previous output
                for d in 0..self.head_dim {
                    o[i * self.head_dim + d] *= alpha;
                }

                // Add contribution from this block
                for j in tr_start..tr_end {
                    let attn_val = qk_block[i * (tr_end - tr_start) + (j - tr_start)];
                    if attn_val != f32::NEG_INFINITY {
                        let attn_weight = ((attn_val - new_m).exp()) / new_l;
                        for d in 0..self.head_dim {
                            o[i * self.head_dim + d] += attn_weight * v[j * self.head_dim + d];
                        }
                    }
                }

                // Update stored statistics
                m[i] = new_m;
                l[i] = new_l;
            }
        }

        // Copy output
        output.copy_from_slice(&o);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_config() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.block_size_qk, 128);
        assert!(config.causal);
    }

    #[test]
    fn test_flash_attention_creation() {
        let fa = FlashAttention::new(32, 128);
        assert_eq!(fa.num_heads, 32);
        assert_eq!(fa.head_dim, 128);
    }

    #[test]
    fn test_flash_attention_custom_config() {
        let config = FlashAttentionConfig {
            block_size_qk: 64,
            block_size_soft: 64,
            causal: false,
        };

        let fa = FlashAttention::new(16, 64).with_config(config);
        assert_eq!(fa.config.block_size_qk, 64);
        assert_eq!(fa.config.causal, false);
    }

    #[test]
    fn test_flash_attention_forward() {
        let fa = FlashAttention::new(2, 4);

        // Create test tensors [batch=1, seq=2, heads=2, head_dim=4]
        let q_data = vec![1.0f32; 1 * 2 * 2 * 4];
        let k_data = vec![2.0f32; 1 * 2 * 2 * 4];
        let v_data = vec![3.0f32; 1 * 2 * 2 * 4];

        let q_shape = Shape::new(&[1, 2, 2, 4]);
        let k_shape = Shape::new(&[1, 2, 2, 4]);
        let v_shape = Shape::new(&[1, 2, 2, 4]);

        let q = Tensor::new(None, TensorType::F32, q_shape, TensorData::F32(q_data)).unwrap();
        let k = Tensor::new(None, TensorType::F32, k_shape, TensorData::F32(k_data)).unwrap();
        let v = Tensor::new(None, TensorType::F32, v_shape, TensorData::F32(v_data)).unwrap();

        let result = fa.forward(&q, &k, &v);

        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims, &[1, 2, 2, 4]);
    }
}
