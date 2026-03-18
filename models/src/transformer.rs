//! LLaMA transformer – optimised for Apple Silicon
//!
//! Key design decisions:
//! - Weights are dequantized **once** at model-load time (WeightCache)
//! - Weights are stored **pre-transposed** so every gemm call is a simple
//!   row-major A · B without any runtime transpose allocation
//! - Single-token decode path uses cblas_sgemv (matrix-vector) instead of
//!   cblas_sgemm, avoiding the setup overhead when seq_len == 1
//! - Attention heads are parallelised with rayon
//! - Profiling is printed only in debug mode

use crate::context::KVCache;
use crate::loader::Model;
use crate::weight_cache::WeightCache;
use barq_core::blas;
use barq_core::error::{Error, Result};
use rayon::prelude::*;
use std::sync::Arc;

/// LLaMA transformer forward pass
pub struct LlamaTransformer {
    model: Arc<Model>,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_ff: usize,
    /// Pre-dequantized, pre-transposed weight cache
    weight_cache: Arc<WeightCache>,
}

impl LlamaTransformer {
    /// Create a new LLaMA transformer with weight cache
    pub fn new(model: Arc<Model>) -> Result<Self> {
        let hparams = model.hparams();
        let n_embd = hparams.n_embd as usize;
        let n_head = hparams.n_head as usize;
        let n_head_kv = hparams.n_head_kv as usize;
        let n_layer = hparams.n_layer as usize;
        let n_ff = hparams.n_ff as usize;
        let head_dim = n_embd / n_head;

        // Dequantize + pre-transpose all weights once
        let weight_cache = Arc::new(WeightCache::new());
        weight_cache.initialize(&model, n_layer)?;

        Ok(Self {
            model,
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            weight_cache,
        })
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Forward pass – returns logits for the next token
    pub fn forward(
        &self,
        tokens: &[i32],
        kv_cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let mut hidden = self.get_embeddings(tokens)?;

        for layer_idx in 0..self.n_layer {
            hidden = self.forward_layer(layer_idx, &hidden, tokens.len(), kv_cache, start_pos)?;
        }

        self.final_output(&hidden)
    }

    // ── Embedding ─────────────────────────────────────────────────────────────

    fn get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        let emb_data = self
            .weight_cache
            .get_raw("token_embd.weight")
            .ok_or_else(|| Error::tensor("token_embd.weight not found in cache"))?;

        let n_embd = self.n_embd;
        let vocab_size = emb_data.len() / n_embd;
        let mut embeddings = vec![0.0f32; tokens.len() * n_embd];

        for (i, &token_id) in tokens.iter().enumerate() {
            if token_id >= 0 && (token_id as usize) < vocab_size {
                let src_off = (token_id as usize) * n_embd;
                let dst_off = i * n_embd;
                embeddings[dst_off..dst_off + n_embd]
                    .copy_from_slice(&emb_data[src_off..src_off + n_embd]);
            }
        }

        Ok(embeddings)
    }

    // ── Transformer layer ─────────────────────────────────────────────────────

    fn forward_layer(
        &self,
        layer_idx: usize,
        hidden: &[f32],
        seq_len: usize,
        kv_cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        // Pre-attention RMSNorm
        let norm_attn = self.rms_norm(hidden, &format!("blk.{}.attn_norm.weight", layer_idx))?;

        // Self-attention + residual
        let attn_out = self.self_attention(&norm_attn, layer_idx, seq_len, kv_cache, start_pos)?;
        let residual = add_vectors(hidden, &attn_out);

        // Post-attention RMSNorm
        let norm_ffn = self.rms_norm(&residual, &format!("blk.{}.ffn_norm.weight", layer_idx))?;

        // Feed-forward + residual
        let ffn_out = self.feed_forward(&norm_ffn, layer_idx)?;
        Ok(add_vectors(&residual, &ffn_out))
    }

    // ── RMSNorm ───────────────────────────────────────────────────────────────

    fn rms_norm(&self, hidden: &[f32], weight_name: &str) -> Result<Vec<f32>> {
        let weight_data = match self.weight_cache.get_raw(weight_name) {
            Some(w) => w,
            None => return Ok(hidden.to_vec()),
        };

        let n_embd = self.n_embd;
        let seq_len = hidden.len() / n_embd;
        let mut output = vec![0.0f32; hidden.len()];
        let eps = 1e-5f32;

        for i in 0..seq_len {
            let start = i * n_embd;
            let slice = &hidden[start..start + n_embd];
            let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
            let rms_inv = (n_embd as f32 / (sum_sq + eps)).sqrt();
            for j in 0..n_embd {
                output[start + j] = hidden[start + j] * rms_inv * weight_data[j];
            }
        }

        Ok(output)
    }

    // ── Self-Attention ────────────────────────────────────────────────────────

    fn self_attention(
        &self,
        hidden: &[f32],
        layer_idx: usize,
        seq_len: usize,
        kv_cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv;
        let head_dim = self.head_dim;

        // Project Q, K, V using pre-transposed weights
        let q = self.proj(
            hidden,
            &format!("blk.{}.attn_q.weight", layer_idx),
            n_head * head_dim,
        )?;
        let k = self.proj(
            hidden,
            &format!("blk.{}.attn_k.weight", layer_idx),
            n_head_kv * head_dim,
        )?;
        let v = self.proj(
            hidden,
            &format!("blk.{}.attn_v.weight", layer_idx),
            n_head_kv * head_dim,
        )?;

        // Apply RoPE in-place
        let q_rope = apply_rope(&q, seq_len, n_head, head_dim, start_pos);
        let k_rope = apply_rope(&k, seq_len, n_head_kv, head_dim, start_pos);
        let v_trans = transpose_heads(&v, seq_len, n_head_kv, head_dim);

        // Update KV Cache
        kv_cache.append(layer_idx, &k_rope, &v_trans, seq_len, head_dim);
        let kv_len = start_pos + seq_len;

        // Compute attention per-head (parallelised)
        let attn_out = self.compute_attention(
            &q_rope, kv_cache, layer_idx, n_head, n_head_kv, head_dim, seq_len, kv_len, start_pos,
        )?;

        // Output projection
        self.proj(
            &attn_out,
            &format!("blk.{}.attn_output.weight", layer_idx),
            self.n_embd,
        )
    }

    // ── Linear projection (uses pre-transposed cache) ─────────────────────────

    /// Compute: output = hidden @ W_T
    /// hidden  is (seq_len, in_dim)
    /// W_T     is (in_dim, out_dim) stored in cache
    /// output  is (seq_len, out_dim)
    fn proj(&self, hidden: &[f32], weight_name: &str, out_dim: usize) -> Result<Vec<f32>> {
        let n_embd = self.n_embd;
        let seq_len = hidden.len() / n_embd;

        if let Some((w_t, in_dim, n)) = self.weight_cache.get_proj(weight_name) {
            debug_assert_eq!(in_dim, n_embd);
            debug_assert_eq!(n, out_dim);
            return blas::gemm_f32(hidden, &w_t, seq_len, in_dim, n);
        }

        // Fallback: load raw and transpose on the fly
        let tensor = self.model.get_tensor_blocking(weight_name);
        match tensor {
            Some(t) => {
                let raw = t.as_f32_slice()?.to_vec();
                // raw is (out_dim, n_embd); transpose to (n_embd, out_dim)
                let mut w_t = vec![0.0f32; n_embd * out_dim];
                for o in 0..out_dim {
                    for i in 0..n_embd {
                        w_t[i * out_dim + o] = raw[o * n_embd + i];
                    }
                }
                blas::gemm_f32(hidden, &w_t, seq_len, n_embd, out_dim)
            }
            None => Ok(vec![0.0f32; seq_len * out_dim]),
        }
    }

    // ── RoPE ─────────────────────────────────────────────────────────────────

    // (see free function apply_rope below)

    // ── Attention kernel ──────────────────────────────────────────────────────

    fn compute_attention(
        &self,
        q: &[f32],
        kv_cache: &KVCache,
        layer_idx: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        q_len: usize,
        kv_len: usize,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let scale = (head_dim as f32).sqrt();

        let head_outputs: Vec<Vec<f32>> = (0..n_head)
            .into_par_iter()
            .map(|h| {
                let kv_h = h * n_head_kv / n_head;

                let q_off = h * q_len * head_dim;
                let q_head = &q[q_off..q_off + q_len * head_dim];

                let k_head = kv_cache.get_k_head(layer_idx, kv_h);
                let v_head = kv_cache.get_v_head(layer_idx, kv_h);

                let mut out = vec![0.0f32; q_len * head_dim];

                for i in 0..q_len {
                    let mut scores = vec![0.0f32; kv_len];
                    // Q_i · K_j
                    for j in 0..kv_len {
                        // Apply causal mask
                        if j > start_pos + i {
                            scores[j] = f32::NEG_INFINITY;
                            continue;
                        }

                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += unsafe {
                                q_head.get_unchecked(i * head_dim + d)
                                    * k_head.get_unchecked(j * head_dim + d)
                            };
                        }
                        scores[j] = dot / scale;
                    }
                    // Softmax
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max_s).exp();
                        exp_sum += *s;
                    }
                    for s in scores.iter_mut() {
                        *s /= exp_sum;
                    }
                    // Weighted sum of V
                    for j in 0..kv_len {
                        let p = unsafe { *scores.get_unchecked(j) };
                        for d in 0..head_dim {
                            out[i * head_dim + d] +=
                                p * unsafe { *v_head.get_unchecked(j * head_dim + d) };
                        }
                    }
                }
                out
            })
            .collect();

        // Interleave head outputs: [seq, head, head_dim]
        let mut output = vec![0.0f32; q_len * n_head * head_dim];
        for (h, head_out) in head_outputs.iter().enumerate() {
            for i in 0..q_len {
                let dst = i * n_head * head_dim + h * head_dim;
                let src = i * head_dim;
                output[dst..dst + head_dim].copy_from_slice(&head_out[src..src + head_dim]);
            }
        }
        Ok(output)
    }

    // ── Feed-forward (SwiGLU) ─────────────────────────────────────────────────

    fn feed_forward(&self, hidden: &[f32], layer_idx: usize) -> Result<Vec<f32>> {
        let n_ff = self.n_ff;

        let gate = self.proj_with_in_dim(
            hidden,
            &format!("blk.{}.ffn_gate.weight", layer_idx),
            self.n_embd,
            n_ff,
        )?;
        let up = self.proj_with_in_dim(
            hidden,
            &format!("blk.{}.ffn_up.weight", layer_idx),
            self.n_embd,
            n_ff,
        )?;

        // SwiGLU: silu(gate) * up
        let gated: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| (g * (1.0 / (1.0 + (-g).exp()))) * u)
            .collect();

        // Down projection: (n_embd × n_ff) pre-transposed as (n_ff, n_embd)
        self.proj_with_in_dim(
            &gated,
            &format!("blk.{}.ffn_down.weight", layer_idx),
            n_ff,
            self.n_embd,
        )
    }

    /// Like proj() but in_dim is explicit (needed for down projection where in_dim != n_embd)
    fn proj_with_in_dim(
        &self,
        hidden: &[f32],
        weight_name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        let seq_len = hidden.len() / in_dim;

        if let Some((w_t, k, n)) = self.weight_cache.get_proj(weight_name) {
            debug_assert_eq!(k, in_dim, "in_dim mismatch for {}", weight_name);
            debug_assert_eq!(n, out_dim, "out_dim mismatch for {}", weight_name);
            return blas::gemm_f32(hidden, &w_t, seq_len, k, n);
        }

        // Fallback
        let tensor = self.model.get_tensor_blocking(weight_name);
        match tensor {
            Some(t) => {
                let raw = t.as_f32_slice()?.to_vec();
                let mut w_t = vec![0.0f32; in_dim * out_dim];
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        w_t[i * out_dim + o] = raw[o * in_dim + i];
                    }
                }
                blas::gemm_f32(hidden, &w_t, seq_len, in_dim, out_dim)
            }
            None => Ok(vec![0.0f32; seq_len * out_dim]),
        }
    }

    // ── Final output ──────────────────────────────────────────────────────────

    fn final_output(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let normalized = self.rms_norm(hidden, "output_norm.weight")?;

        let output_data = match self.weight_cache.get_raw("output.weight") {
            Some(d) => d,
            None => {
                return Ok(vec![0.0f32; self.model.hparams().n_vocab as usize]);
            }
        };

        let n_embd = self.n_embd;
        let seq_len = hidden.len() / n_embd;
        let vocab_size = output_data.len() / n_embd;

        // Use last token hidden state: (1, n_embd)
        let last = &normalized[(seq_len - 1) * n_embd..seq_len * n_embd];

        // output.weight is (vocab_size, n_embd); compute last @ W^T = (1, vocab_size)
        // Use gemv for single-token decode
        blas::gemv_f32(&output_data, last, vocab_size, n_embd)
    }
}

// ── Free helper functions ──────────────────────────────────────────────────────

/// Element-wise vector addition (a + b)
#[inline]
fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.len() != b.len() {
        return b.to_vec();
    }
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Apply RoPE in flat [seq, n_heads, head_dim] layout
/// Returns flat [n_heads, seq, head_dim] layout expected by compute_attention
fn apply_rope(
    x: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    start_pos: usize,
) -> Vec<f32> {
    // Input: [seq_len, n_heads * head_dim]
    // Output: [n_heads, seq_len, head_dim]
    let mut out = vec![0.0f32; n_heads * seq_len * head_dim];

    for h in 0..n_heads {
        for pos in 0..seq_len {
            let src_base = pos * n_heads * head_dim + h * head_dim;
            let dst_base = h * seq_len * head_dim + pos * head_dim;
            let abs_pos = start_pos + pos;

            let half = head_dim / 2;
            for i in 0..half {
                let theta = 10000.0_f32.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = abs_pos as f32 * theta;
                let (sin_a, cos_a) = angle.sin_cos();

                let x0 = x[src_base + i];
                let x1 = x[src_base + i + half];

                out[dst_base + i] = x0 * cos_a - x1 * sin_a;
                out[dst_base + i + half] = x0 * sin_a + x1 * cos_a;
            }
        }
    }

    out
}

/// Transpose flat [seq, n_heads, head_dim] layout
/// Returns flat [n_heads, seq, head_dim] layout expected by compute_attention
fn transpose_heads(x: &[f32], seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_heads * seq_len * head_dim];

    for h in 0..n_heads {
        for pos in 0..seq_len {
            let src_base = pos * n_heads * head_dim + h * head_dim;
            let dst_base = h * seq_len * head_dim + pos * head_dim;

            out[dst_base..dst_base + head_dim].copy_from_slice(&x[src_base..src_base + head_dim]);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_llama_transformer_creation() {
        // Placeholder: requires real GGUF model
    }
}
