//! Model inference context
//!
//! Manages the inference state including KV cache and processing.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::loader::Model;
use barq_core::error::{Error, Result};
use barq_core::tensor::Tensor;

/// Inference context parameters
#[derive(Debug, Clone)]
pub struct ContextParams {
    /// Context size (0 = use model default)
    pub n_ctx: u32,
    /// Batch size
    pub n_batch: u32,
    /// Number of threads for generation (GPU inference: use 4, CPU: use physical cores)
    pub n_threads: u32,
    /// Number of threads for batch processing
    pub n_threads_batch: u32,
    /// Rope scaling type
    pub rope_scaling_type: u32,
    /// Rope frequency base (0 = use model default)
    pub rope_freq_base: f32,
    /// Rope frequency scale (0 = use model default)
    pub rope_freq_scale: f32,
    /// Enable embeddings extraction
    pub embeddings: bool,

    // === Performance Optimizations ===
    /// Enable Flash Attention (reduces memory bandwidth, ~30% faster)
    pub flash_attn: bool,
    /// Number of GPU layers to offload (9999 = all layers)
    pub n_gpu_layers: i32,
    /// KV cache quantization type (Q8_0 = 50% VRAM savings, Q4_0 = 75% savings)
    pub type_k: u32, // GGML_TYPE_Q8_0 or GGML_TYPE_Q4_0
    pub type_v: u32,
    /// KV cache defragmentation threshold (0.0-1.0, default 0.1 = 10%)
    pub defrag_thold: f32,
    /// Micro batch size (should match n_batch for better Flash Attention utilization)
    pub n_ubatch: u32,
    /// Maximum sequences in batch (for continuous batching)
    pub n_seq_max: u32,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            // Standard parameters
            n_ctx: 8192, // Larger context safe with Flash Attention
            n_batch: 512,
            n_threads: 4, // Optimized for GPU inference
            n_threads_batch: 4,
            rope_scaling_type: 0,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            embeddings: false,

            // Performance optimizations (Phase 1: Zero-Cost Flag Wins)
            flash_attn: true,   // ~30% faster speculative decoding
            n_gpu_layers: 9999, // Offload all layers to GPU
            type_k: 1,          // GGML_TYPE_Q8_0 (50% VRAM savings)
            type_v: 1,          // GGML_TYPE_Q8_0
            defrag_thold: 0.1,  // Defrag at 10% fragmentation
            n_ubatch: 512,      // Match n_batch for Flash Attention
            n_seq_max: 32,      // Support continuous batching
        }
    }
}

impl ContextParams {
    /// Create optimized parameters for GPU inference
    pub fn gpu_optimized() -> Self {
        Self {
            n_threads: 4, // CPU is bottleneck, not compute
            n_threads_batch: 4,
            flash_attn: true,
            n_gpu_layers: 9999,
            type_k: 1, // Q8_0
            type_v: 1,
            n_ctx: 8192,
            ..Default::default()
        }
    }

    /// Create optimized parameters for CPU inference
    pub fn cpu_optimized() -> Self {
        Self {
            n_threads: num_cpus::get_physical() as u32,
            n_threads_batch: num_cpus::get_physical() as u32,
            flash_attn: false, // CPU Flash Attention less beneficial
            n_gpu_layers: 0,
            type_k: 1,
            type_v: 1,
            ..Default::default()
        }
    }

    /// Create parameters for maximum quality (Q8_0 quantization)
    pub fn quality() -> Self {
        Self {
            type_k: 1, // Q8_0
            type_v: 1,
            flash_attn: true,
            ..Default::default()
        }
    }

    /// Create parameters for maximum speed (IQ4_XS quantization)
    pub fn speed() -> Self {
        Self {
            type_k: 0, // Q4_0 for even more compression
            type_v: 0,
            flash_attn: true,
            n_gpu_layers: 9999,
            n_ctx: 4096, // Smaller context for speed
            ..Default::default()
        }
    }
}

/// Token batch for processing
#[derive(Debug, Clone)]
pub struct Batch {
    /// Number of tokens in batch
    pub n_tokens: i32,
    /// Token IDs
    pub token: Vec<i32>,
    /// Embeddings (optional)
    pub embd: Option<Vec<f32>>,
    /// Positions
    pub pos: Vec<i32>,
    /// Sequence IDs
    pub n_seq_id: Vec<i32>,
    pub seq_id: Vec<Vec<i32>>,
    /// Output logits
    pub logits: Vec<i8>,
}

impl Batch {
    /// Create a single token batch
    pub fn single(token: i32) -> Self {
        Self {
            n_tokens: 1,
            token: vec![token],
            embd: None,
            pos: vec![0],
            n_seq_id: vec![1],
            seq_id: vec![vec![0]],
            logits: vec![1],
        }
    }

    /// Create a batch from token IDs
    pub fn from_tokens(tokens: &[i32]) -> Self {
        let n_tokens = tokens.len() as i32;
        Self {
            n_tokens,
            token: tokens.to_vec(),
            embd: None,
            pos: (0..n_tokens).collect(),
            n_seq_id: vec![1],
            seq_id: vec![vec![0]; n_tokens as usize],
            logits: vec![0; n_tokens as usize - 1],
        }
    }
}

/// Inference context
pub struct ModelContext {
    /// Model reference
    model: Arc<Model>,
    /// Context parameters
    params: ContextParams,
    /// KV cache
    kv_cache: Arc<Mutex<KVCache>>,
    /// Current position
    pos: Arc<Mutex<usize>>,
    /// Transformer (holds dequantised weight cache – create once)
    transformer: Arc<crate::transformer::LlamaTransformer>,
}

/// KV cache for attention
/// KV cache for attention
pub struct KVCache {
    /// K cache per layer. Layout: [layer][head_kv] -> flattened sequence data
    k_cache: Vec<Vec<Vec<f32>>>,
    /// V cache per layer. Layout: [layer][head_kv] -> flattened sequence data
    v_cache: Vec<Vec<Vec<f32>>>,
    /// Maximum sequence length
    max_size: usize,
    /// Current number of tokens in cache
    size: usize,
    /// Number of KV heads
    n_head_kv: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(max_size: usize, n_layers: usize, n_head_kv: usize) -> Self {
        Self {
            k_cache: vec![vec![Vec::new(); n_head_kv]; n_layers],
            v_cache: vec![vec![Vec::new(); n_head_kv]; n_layers],
            max_size,
            size: 0,
            n_head_kv,
        }
    }

    /// Append new K and V tokens for a specific layer.
    /// Both `k` and `v` must be in [n_head_kv, seq_len, head_dim] flattened layout
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32], n_tokens: usize, head_dim: usize) {
        if layer >= self.k_cache.len() {
            return;
        }

        // We must append head by head to maintain contiguous memory per head!
        for h in 0..self.n_head_kv {
            let start = h * n_tokens * head_dim;
            let end = (h + 1) * n_tokens * head_dim;
            
            self.k_cache[layer][h].extend_from_slice(&k[start..end]);
            self.v_cache[layer][h].extend_from_slice(&v[start..end]);
        }

        // Only layer 0 updates the total cache size
        if layer == 0 {
            self.size += n_tokens;
        }
    }

    /// Get current cache size (start_pos for new tokens)
    pub fn size(&self) -> usize {
        self.size
    }

    /// Retrieve full K cache for a layer, flattened as [n_head_kv, total_seq_len, head_dim]
    pub fn get_k_flattened(&self, layer: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_head_kv * self.size * 128); // rough capacity
        for h in 0..self.n_head_kv {
            out.extend_from_slice(&self.k_cache[layer][h]);
        }
        out
    }

    /// Retrieve full V cache for a layer, flattened as [n_head_kv, total_seq_len, head_dim]
    pub fn get_v_flattened(&self, layer: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_head_kv * self.size * 128); // rough capacity
        for h in 0..self.n_head_kv {
            out.extend_from_slice(&self.v_cache[layer][h]);
        }
        out
    }

    /// Get reference to K cache for a specific layer and head. Shape: [total_seq_len * head_dim]
    pub fn get_k_head(&self, layer: usize, head: usize) -> &[f32] {
        &self.k_cache[layer][head]
    }

    /// Get reference to V cache for a specific layer and head. Shape: [total_seq_len * head_dim]
    pub fn get_v_head(&self, layer: usize, head: usize) -> &[f32] {
        &self.v_cache[layer][head]
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for layer_k in &mut self.k_cache {
            for h in layer_k {
                h.clear();
            }
        }
        for layer_v in &mut self.v_cache {
            for h in layer_v {
                h.clear();
            }
        }
        self.size = 0;
    }
}

impl ModelContext {
    /// Create a new inference context
    pub fn new(model: Arc<Model>, params: ContextParams) -> Result<Self> {
        let n_ctx = if params.n_ctx == 0 {
            model.hparams.n_ctx_train as usize
        } else {
            params.n_ctx as usize
        };

        let n_layer = model.hparams.n_layer as usize;
        let n_head_kv = model.hparams.n_head_kv as usize;
        let kv_cache = KVCache::new(n_ctx, n_layer, n_head_kv);

        // Build transformer + weight cache ONCE
        let transformer = Arc::new(crate::transformer::LlamaTransformer::new(model.clone())?);

        Ok(Self {
            model,
            params,
            kv_cache: Arc::new(Mutex::new(kv_cache)),
            pos: Arc::new(Mutex::new(0)),
            transformer,
        })
    }

    /// Create context with default parameters
    pub fn with_model(model: Arc<Model>) -> Result<Self> {
        Self::new(model, ContextParams::default())
    }

    /// Encode a batch of tokens (no KV cache)
    pub async fn encode(&self, batch: &Batch) -> Result<Vec<f32>> {
        let mut cache = self.kv_cache.lock().await;
        // Ensure cache starts empty for new encode
        cache.clear();
        
        let mut pos = self.pos.lock().await;
        *pos = 0;
        
        let logits = self.transformer.forward(&batch.token, &mut cache, *pos)?;

        *pos += batch.n_tokens as usize;
        
        Ok(logits)
    }

    /// Decode a batch of tokens (uses KV cache)
    pub async fn decode(&self, batch: &Batch) -> Result<Vec<f32>> {
        let mut pos = self.pos.lock().await;
        let mut cache = self.kv_cache.lock().await;

        let logits = self.transformer.forward(&batch.token, &mut cache, *pos)?;

        *pos += batch.n_tokens as usize;
        Ok(logits)
    }

    /// Sample a token from logits
    pub fn sample(&self, logits: &[f32], temperature: f32, top_k: i32, top_p: f32) -> Result<i32> {
        // Check for NaN or empty logits
        if logits.is_empty() {
            return Err(Error::tensor("Empty logits"));
        }

        // Filter out NaN values
        let valid_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .filter(|&(_, &logit)| !logit.is_nan())
            .map(|(i, &logit)| (i, logit))
            .collect();

        if valid_logits.is_empty() {
            return Err(Error::tensor("All logits are NaN"));
        }

        let mut token_data = valid_logits;

        // Sort by logit value (descending)
        token_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply temperature
        if temperature > 0.0 && temperature != 1.0 {
            for (_, logit) in token_data.iter_mut() {
                *logit /= temperature;
            }
        }

        // Apply top-k sampling
        if top_k > 0 && top_k < token_data.len() as i32 {
            token_data.truncate(top_k as usize);
        }

        // Apply top-p (nucleus) sampling
        if top_p < 1.0 {
            // Compute softmax probabilities
            let max_logit = token_data.first().map(|&(_, l)| l).unwrap_or(0.0f32);
            let exp_sum: f32 = token_data.iter().map(|&(_, l)| (l - max_logit).exp()).sum();

            let mut cumulative = 0.0f32;
            let mut cutoff_idx = token_data.len();

            for (i, &(_, logit)) in token_data.iter().enumerate() {
                let prob = (logit - max_logit).exp() / exp_sum;
                cumulative += prob;
                if cumulative >= top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            token_data.truncate(cutoff_idx);
        }

        // Sample from remaining tokens
        if token_data.is_empty() {
            // Fallback to argmax over all tokens
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as i32)
                .unwrap());
        }

        // Compute final probabilities
        let max_logit = token_data.first().map(|&(_, l)| l).unwrap_or(0.0f32);
        let exp_sum: f32 = token_data.iter().map(|&(_, l)| (l - max_logit).exp()).sum();

        // Sample
        let r: f32 = rand::random();
        let mut cumulative = 0.0f32;

        for (idx, &(_, logit)) in token_data.iter().enumerate() {
            let prob = (logit - max_logit).exp() / exp_sum;
            cumulative += prob;
            if r <= cumulative {
                return Ok(token_data[idx].0 as i32);
            }
        }

        // Fallback to last token
        Ok(token_data.last().map(|&(i, _)| i as i32).unwrap())
    }

    /// Generate tokens
    pub async fn generate(
        &self,
        tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
    ) -> Result<Vec<i32>> {
        let mut output = Vec::new();
        let mut current_tokens = tokens.to_vec();

        // Process prompt
        let batch = Batch::from_tokens(&current_tokens);
        let mut logits = self.encode(&batch).await?;

        // Generate tokens
        for _ in 0..max_tokens {
            let token = self.sample(&logits, temperature, top_k, top_p)?;
            output.push(token);
            current_tokens.push(token);

            // Feed new token to get next logits
            let batch = Batch::single(token);
            logits = self.decode(&batch).await?;

            // Check for EOS
            if token == 0 || token == 2 {
                break;
            }
        }

        Ok(output)
    }

    /// Clear the KV cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.kv_cache.lock().await;
        cache.clear();
        *self.pos.lock().await = 0;
        Ok(())
    }

    /// Returns the current position
    pub async fn pos(&self) -> usize {
        *self.pos.lock().await
    }

    /// Returns the model
    pub fn model(&self) -> &Model {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kv_cache() {
        let cache = KVCache::new(2048, 32);
        assert_eq!(cache.size(), 0);

        assert!(cache.get(0, true).is_none());
        assert!(cache.get(0, false).is_none());
    }

    #[tokio::test]
    async fn test_batch() {
        let batch = Batch::single(42);
        assert_eq!(batch.n_tokens, 1);
        assert_eq!(batch.token[0], 42);
    }

    #[tokio::test]
    async fn test_context_params() {
        let params = ContextParams::default();
        assert_eq!(params.n_threads, 4);
        assert_eq!(params.n_batch, 512);
    }
}
