//! Multi-head attention implementation

use core::tensor::{Tensor, TensorType, Shape, TensorData};
use core::error::{Error, Result};
use core::softmax;
use core::normalization;

/// Attention configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub n_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Whether to use causal masking
    pub causal: bool,
}

impl AttentionConfig {
    pub fn new(n_heads: usize, head_dim: usize, causal: bool) -> Self {
        Self {
            n_heads,
            head_dim,
            causal,
        }
    }
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            n_heads: 32,
            head_dim: 128,
            causal: true,
        }
    }
}

/// Multi-head attention
pub struct MultiHeadAttention {
    config: AttentionConfig,
}

impl MultiHeadAttention {
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    /// Compute multi-head attention
    ///
    /// Input shapes:
    /// - q: [batch_size, n_heads, seq_len, head_dim]
    /// - k: [batch_size, n_heads, seq_len, head_dim]
    /// - v: [batch_size, n_heads, seq_len, head_dim]
    ///
    /// Output shape: [batch_size, seq_len, n_heads * head_dim]
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor> {
        if q.dtype() != TensorType::F32 {
            return Err(Error::Unsupported("Attention requires f32 tensors".to_string()));
        }

        let q_shape = q.shape().dims();
        let batch_size = q_shape[0];
        let seq_len = q_shape[2];

        // TODO: Implement actual multi-head attention
        // For now, return dummy output with correct shape

        let output_shape = Shape::new(vec![
            batch_size,
            seq_len,
            self.config.n_heads * self.config.head_dim,
        ]);

        let output_data = vec![0.0f32; output_shape.num_elements()];

        Tensor::new(
            Some("attention_output".to_string()),
            TensorType::F32,
            output_shape,
            TensorData::F32(output_data),
        )
    }

    /// Compute scaled dot-product attention
    pub fn scaled_dot_product_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<Vec<f32>> {
        let head_dim = self.config.head_dim;
        let seq_len = q.len() / head_dim;

        // Compute QK^T / sqrt(d_k)
        let scale = (head_dim as f32).sqrt();

        let mut output = Vec::with_capacity(v.len());

        for i in 0..seq_len {
            let mut attn_weights = Vec::with_capacity(seq_len);

            for j in 0..seq_len {
                if self.config.causal && j > i {
                    // Causal masking: future positions are masked
                    attn_weights.push(f32::NEG_INFINITY);
                    continue;
                }

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }

                attn_weights.push(dot / scale);
            }

            // Apply softmax
            let attn_probs = softmax(&attn_weights)?;

            // Compute weighted sum of values
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for j in 0..seq_len {
                    if !self.config.causal || j <= i {
                        sum += attn_probs[j] * v[j * head_dim + d];
                    }
                }
                output.push(sum);
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::new(32, 128, true);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert!(config.causal);
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let config = AttentionConfig::new(4, 8, false);
        let attn = MultiHeadAttention::new(config);

        let head_dim = 8;
        let seq_len = 4;

        // Create dummy Q, K, V
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();

        let result = attn.scaled_dot_product_attention(&q, &k, &v).unwrap();

        assert_eq!(result.len(), seq_len * head_dim);
    }

    #[test]
    fn test_scaled_dot_product_attention_causal() {
        let config = AttentionConfig::new(4, 8, true);
        let attn = MultiHeadAttention::new(config);

        let head_dim = 8;
        let seq_len = 4;

        let q: Vec<f32> = vec![1.0; seq_len * head_dim];
        let k: Vec<f32> = vec![1.0; seq_len * head_dim];
        let v: Vec<f32> = vec![1.0; seq_len * head_dim];

        let result = attn.scaled_dot_product_attention(&q, &k, &v).unwrap();

        assert_eq!(result.len(), seq_len * head_dim);

        // All results should be equal since all inputs are the same
        let first = result[0];
        for &val in &result {
            assert!((val - first).abs() < 1e-4);
        }
    }
}
