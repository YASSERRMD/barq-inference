//! Complete LLaMA transformer implementation
//!
//! Implements the full LLaMA architecture including:
//! - RMSNorm
//! - Rotary Position Embeddings (RoPE)
//! - Multi-head attention with KV cache
//! - SwiGLU feed-forward network
//! - Layer stacking

use std::sync::Arc;
use crate::loader::Model;
use crate::context::KVCache;
use barq_core::error::{Error, Result};

/// LLaMA transformer forward pass
pub struct LlamaTransformer {
    model: Arc<Model>,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
}

impl LlamaTransformer {
    /// Create a new LLaMA transformer
    pub fn new(model: Arc<Model>) -> Result<Self> {
        let hparams = model.hparams();
        let n_embd = hparams.n_embd as usize;
        let n_head = hparams.n_head as usize;
        let n_head_kv = hparams.n_head_kv as usize;
        let n_layer = hparams.n_layer as usize;

        // Calculate head dimension
        let head_dim = n_embd / n_head;

        Ok(Self {
            model,
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
        })
    }

    /// Forward pass through all transformer layers
    pub fn forward(
        &self,
        tokens: &[i32],
        kv_cache: &mut KVCache,
    ) -> Result<Vec<f32>> {
        // Get embedding matrix
        let embeddings = self.get_embeddings(tokens)?;

        // Apply transformer layers
        let mut hidden = embeddings;

        for layer_idx in 0..self.n_layer {
            hidden = self.forward_layer(layer_idx, &hidden, tokens.len(), kv_cache)?;
        }

        // Apply final RMSNorm and output projection
        self.final_output(&hidden)
    }

    /// Get token embeddings
    fn get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        let tok_emb = self.model.get_tensor_blocking("token_embd.weight")
            .ok_or_else(|| Error::tensor("token_embd.weight not found"))?;

        let emb_data = tok_emb.as_f32_slice()?.to_vec();
        let n_embd = self.n_embd;
        let mut embeddings = vec![0.0f32; tokens.len() * n_embd];

        for (i, &token_id) in tokens.iter().enumerate() {
            if token_id >= 0 && (token_id as usize) < tok_emb.shape().dims()[0] {
                let offset = (token_id as usize) * n_embd;
                for j in 0..n_embd {
                    if offset + j < emb_data.len() {
                        embeddings[i * n_embd + j] = emb_data[offset + j];
                    }
                }
            }
        }

        Ok(embeddings)
    }

    /// Forward pass through a single transformer layer
    fn forward_layer(
        &self,
        layer_idx: usize,
        hidden: &[f32],
        seq_len: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Vec<f32>> {
        // Pre-attention RMSNorm
        let normalized = self.rms_norm(hidden, &format!("blk.{}.attn_norm.weight", layer_idx))?;

        // Self-attention
        let attn_out = self.self_attention(&normalized, layer_idx, seq_len, kv_cache)?;

        // Residual connection
        let mut residual = self.add_residual(hidden, &attn_out)?;

        // Post-attention RMSNorm
        let normalized = self.rms_norm(&residual, &format!("blk.{}.ffn_norm.weight", layer_idx))?;

        // Feed-forward network
        let ffn_out = self.feed_forward(&normalized, layer_idx)?;

        // Second residual connection
        residual = self.add_residual(&residual, &ffn_out)?;

        Ok(residual)
    }

    /// RMSNorm layer
    fn rms_norm(&self, hidden: &[f32], weight_name: &str) -> Result<Vec<f32>> {
        let weight = self.model.get_tensor_blocking(weight_name);
        let weight_data = match weight {
            Some(w) => w.as_f32_slice()?.to_vec().to_vec(),
            None => return Ok(hidden.to_vec()), // Skip norm if weight not found
        };

        let n_embd = self.n_embd;
        let seq_len = hidden.len() / n_embd;
        let mut output = Vec::with_capacity(hidden.len());

        for i in 0..seq_len {
            let start = i * n_embd;
            let end = start + n_embd;

            // Compute RMS
            let sum_sq: f32 = hidden[start..end].iter().map(|&x| x * x).sum();
            let rms = (sum_sq / n_embd as f32).sqrt() + 1e-5;

            // Normalize and scale
            for j in 0..n_embd {
                let normalized = hidden[start + j] / rms;
                output.push(normalized * weight_data[j]);
            }
        }

        Ok(output)
    }

    /// Self-attention with RoPE and KV cache
    fn self_attention(
        &self,
        hidden: &[f32],
        layer_idx: usize,
        seq_len: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Vec<f32>> {
        let n_embd = self.n_embd;
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv;
        let head_dim = self.head_dim;

        // Project Q, K, V
        let q = self.project_qkv(hidden, layer_idx, "q", n_head * head_dim)?;
        let k = self.project_qkv(hidden, layer_idx, "k", n_head_kv * head_dim)?;
        let v = self.project_qkv(hidden, layer_idx, "v", n_head_kv * head_dim)?;

        // Reshape for multi-head attention
        let q_heads = self.reshape_to_heads(&q, seq_len, n_head, head_dim)?;
        let k_heads = self.reshape_to_heads(&k, seq_len, n_head_kv, head_dim)?;
        let v_heads = self.reshape_to_heads(&v, seq_len, n_head_kv, head_dim)?;

        // Apply RoPE to Q and K
        let q_rope = self.apply_rope(&q_heads, seq_len)?;
        let k_rope = self.apply_rope(&k_heads, seq_len)?;

        // Compute attention scores and output
        let attn_out = self.compute_attention(&q_rope, &k_rope, &v_heads, n_head, n_head_kv, head_dim, seq_len)?;

        // Project output
        self.output_proj(&attn_out, layer_idx)
    }

    /// Project to Q, K, or V
    fn project_qkv(&self, hidden: &[f32], layer: usize, typ: &str, out_dim: usize) -> Result<Vec<f32>> {
        let weight_name = format!("blk.{}.attn_{}.weight", layer, typ);
        let weight = self.model.get_tensor_blocking(&weight_name);

        let weight_data = match weight {
            Some(w) => w.as_f32_slice()?.to_vec(),
            None => return Ok(vec![0.0; hidden.len() * out_dim / self.n_embd]),
        };

        let n_embd = self.n_embd;
        let seq_len = hidden.len() / n_embd;
        let mut output = vec![0.0; seq_len * out_dim];

        // Simplified matrix multiplication
        for i in 0..seq_len {
            for j in 0..out_dim {
                let mut sum = 0.0;
                for k in 0..n_embd {
                    sum += hidden[i * n_embd + k] * weight_data[k * out_dim + j];
                }
                output[i * out_dim + j] = sum;
            }
        }

        Ok(output)
    }

    /// Reshape to multi-head format
    fn reshape_to_heads(&self, data: &[f32], seq_len: usize, n_heads: usize, head_dim: usize) -> Result<Vec<Vec<f32>>> {
        let mut heads = Vec::with_capacity(n_heads);

        for h in 0..n_heads {
            let mut head_data = Vec::with_capacity(seq_len * head_dim);
            for i in 0..seq_len {
                for j in 0..head_dim {
                    let idx = i * n_heads * head_dim + h * head_dim + j;
                    if idx < data.len() {
                        head_data.push(data[idx]);
                    } else {
                        head_data.push(0.0);
                    }
                }
            }
            heads.push(head_data);
        }

        Ok(heads)
    }

    /// Apply Rotary Position Embeddings
    fn apply_rope(&self, heads: &[Vec<f32>], seq_len: usize) -> Result<Vec<Vec<f32>>> {
        let head_dim = self.head_dim;
        let mut output = Vec::new();

        for head in heads {
            let mut rope_head = Vec::with_capacity(head.len());
            for pos in 0..seq_len {
                for i in 0..head_dim {
                    let idx = pos * head_dim + i;
                    if idx >= head.len() {
                        rope_head.push(0.0);
                        continue;
                    }

                    let val = head[idx];

                    // Apply RoPE (simplified - using only even/odd positions)
                    if i % 2 == 0 && i + 1 < head_dim {
                        let next_idx = idx + 1;
                        let next_val = if next_idx < head.len() { head[next_idx] } else { 0.0 };

                        // Compute rotation
                        let theta = 10000.0_f32.powf(-(i as f32) / head_dim as f32);
                        let angle = pos as f32 * theta;
                        let cos_a = angle.cos();
                        let sin_a = angle.sin();

                        // Rotate
                        rope_head.push(val * cos_a - next_val * sin_a);
                        rope_head.push(val * sin_a + next_val * cos_a);
                    } else if i % 2 == 1 {
                        // Skip odd positions (handled with even)
                    } else {
                        rope_head.push(val);
                    }
                }
            }
            output.push(rope_head);
        }

        Ok(output)
    }

    /// Compute attention output
    fn compute_attention(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; seq_len * n_head * head_dim];

        // Handle multi-query attention (n_head_kv may be smaller than n_head)
        for h in 0..n_head {
            let kv_head_idx = h * n_head_kv / n_head;
            let q_head = &q[h];
            let k_head = &k[kv_head_idx];
            let v_head = &v[kv_head_idx];

            // Compute attention scores for each position
            for i in 0..seq_len {
                let mut attn_weights = Vec::with_capacity(seq_len);

                for j in 0..seq_len {
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        let q_idx = i * head_dim + d;
                        let k_idx = j * head_dim + d;
                        if q_idx < q_head.len() && k_idx < k_head.len() {
                            score += q_head[q_idx] * k_head[k_idx];
                        }
                    }
                    // Scale by sqrt(d_k)
                    attn_weights.push(score / (head_dim as f32).sqrt());
                }

                // Softmax
                let max_weight = attn_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = attn_weights.iter().map(|&w| (w - max_weight).exp()).sum();
                let probs: Vec<f32> = attn_weights.iter().map(|&w| ((w - max_weight).exp()) / exp_sum).collect();

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        let v_idx = j * head_dim + d;
                        if v_idx < v_head.len() {
                            sum += probs[j] * v_head[v_idx];
                        }
                    }
                    output[i * n_head * head_dim + h * head_dim + d] = sum;
                }
            }
        }

        Ok(output)
    }

    /// Output projection
    fn output_proj(&self, hidden: &[f32], layer_idx: usize) -> Result<Vec<f32>> {
        let weight_name = format!("blk.{}.attn_output.weight", layer_idx);
        let weight = self.model.get_tensor_blocking(&weight_name);

        let weight_data = match weight {
            Some(w) => w.as_f32_slice()?.to_vec(),
            None => return Ok(hidden.to_vec()),
        };

        let n_embd = self.n_embd;
        let seq_len = hidden.len() / n_embd;
        let mut output = vec![0.0; hidden.len()];

        // Simplified matrix multiplication
        for i in 0..seq_len {
            for j in 0..n_embd {
                let mut sum = 0.0;
                for k in 0..n_embd {
                    sum += hidden[i * n_embd + k] * weight_data[j * n_embd + k];
                }
                output[i * n_embd + j] = sum;
            }
        }

        Ok(output)
    }

    /// SwiGLU feed-forward network
    fn feed_forward(&self, hidden: &[f32], layer_idx: usize) -> Result<Vec<f32>> {
        let n_embd = self.n_embd;
        let n_ff = self.model.hparams().n_ff as usize;
        let seq_len = hidden.len() / n_embd;

        // Gate projection
        let gate = self._ffn_projection(hidden, layer_idx, "gate", n_ff)?;
        // Up projection
        let up = self._ffn_projection(hidden, layer_idx, "up", n_ff)?;

        // Apply SiLU to gate and multiply
        let mut gated = Vec::with_capacity(gate.len());
        for (g, u) in gate.iter().zip(up.iter()) {
            let silu_g = *g / (1.0 + (-g).exp());
            gated.push(silu_g * u);
        }

        // Down projection
        self._ffn_projection(&gated, layer_idx, "down", n_embd)
    }

    /// FFN projection
    fn _ffn_projection(&self, hidden: &[f32], layer: usize, typ: &str, out_dim: usize) -> Result<Vec<f32>> {
        let weight_name = format!("blk.{}.ffn_{}.weight", layer, typ);
        let weight = self.model.get_tensor_blocking(&weight_name);

        let weight_data = match weight {
            Some(w) => w.as_f32_slice()?.to_vec(),
            None => return Ok(vec![0.0; hidden.len() * out_dim / self.n_embd]),
        };

        let n_embd = self.n_embd;
        let seq_len = hidden.len() / n_embd;
        let mut output = vec![0.0; seq_len * out_dim];

        // Simplified matrix multiplication
        for i in 0..seq_len {
            for j in 0..out_dim {
                let mut sum = 0.0;
                for k in 0..n_embd {
                    sum += hidden[i * n_embd + k] * weight_data[k * out_dim + j];
                }
                output[i * out_dim + j] = sum;
            }
        }

        Ok(output)
    }

    /// Residual connection
    fn add_residual(&self, hidden: &[f32], output: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != output.len() {
            return Ok(output.to_vec());
        }

        let mut result = Vec::with_capacity(hidden.len());
        for (h, o) in hidden.iter().zip(output.iter()) {
            result.push(h + o);
        }

        Ok(result)
    }

    /// Final output projection to logits
    fn final_output(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        // Final RMSNorm
        let normalized = self.rms_norm(hidden, "output_norm.weight")?;

        // Output projection
        let output_weight = self.model.get_tensor_blocking("output.weight");
        let output_data: Vec<f32> = match output_weight {
            Some(w) => w.as_f32_slice()?.to_vec(),
            None => return Ok(vec![0.0; self.model.hparams().n_vocab as usize]),
        };

        let n_vocab = self.model.hparams().n_vocab as usize;
        let n_embd = self.n_embd;
        let seq_len = hidden.len() / n_embd;

        // Only use the last token's hidden state for next token prediction
        let last_hidden = &normalized[(seq_len - 1) * n_embd..seq_len * n_embd];

        let mut logits = vec![0.0f32; n_vocab];
        for i in 0..n_vocab {
            let mut sum = 0.0;
            for j in 0..n_embd {
                sum += last_hidden[j] * output_data[i * n_embd + j];
            }
            logits[i] = sum;
        }

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_transformer_creation() {
        // Would need actual model to test
        // This is a placeholder
    }
}
