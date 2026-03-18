//! Weight caching system to avoid repeated dequantization
//!
//! Stores dequantized and pre-transposed weights in memory to eliminate
//! the overhead of Q4_K → f32 conversion on every forward pass.
//! Weights are stored in "input-major" layout (K, N) for direct BLAS call.

use barq_core::error::{Error, Result};
use std::collections::HashMap;
use std::sync::RwLock;

/// Cached weight — stored in transposed layout (K, N) for direct BLAS pass
pub struct CachedWeight {
    /// Dequantized f32 data in row-major (K×N) layout
    pub data: Vec<f32>,
    /// rows (K = in_dim)
    pub rows: usize,
    /// cols (N = out_dim)
    pub cols: usize,
}

/// Weight cache for storing pre-dequantized, pre-transposed weights
pub struct WeightCache {
    /// Transposed weights indexed by tensor name
    cache: RwLock<HashMap<String, CachedWeight>>,
}

impl WeightCache {
    /// Create a new weight cache
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize the cache with all required weights in transposed layout
    pub fn initialize(&self, model: &crate::loader::Model, n_layer: usize) -> Result<()> {
        let mut cache = self
            .cache
            .write()
            .map_err(|e| Error::backend(format!("Cache write lock failed: {}", e)))?;

        eprintln!("Initializing weight cache (dequantize + pre-transpose)...");

        let n_embd = model.hparams().n_embd as usize;
        let n_head = model.hparams().n_head as usize;
        let n_head_kv = model.hparams().n_head_kv as usize;
        let n_ff = model.hparams().n_ff as usize;
        let head_dim = n_embd / n_head;

        for layer in 0..n_layer {
            // Q: (n_head * head_dim, n_embd) → transpose to (n_embd, n_head * head_dim)
            Self::cache_transposed(
                &mut cache,
                model,
                &format!("blk.{}.attn_q.weight", layer),
                n_embd,
                n_head * head_dim,
            );
            // K
            Self::cache_transposed(
                &mut cache,
                model,
                &format!("blk.{}.attn_k.weight", layer),
                n_embd,
                n_head_kv * head_dim,
            );
            // V
            Self::cache_transposed(
                &mut cache,
                model,
                &format!("blk.{}.attn_v.weight", layer),
                n_embd,
                n_head_kv * head_dim,
            );
            // Output projection: (n_embd, n_embd)
            Self::cache_transposed(
                &mut cache,
                model,
                &format!("blk.{}.attn_output.weight", layer),
                n_embd,
                n_embd,
            );
            // FFN gate: (n_ff, n_embd) → transpose to (n_embd, n_ff)
            Self::cache_transposed(
                &mut cache,
                model,
                &format!("blk.{}.ffn_gate.weight", layer),
                n_embd,
                n_ff,
            );
            // FFN up: (n_ff, n_embd) → transpose to (n_embd, n_ff)
            Self::cache_transposed(
                &mut cache,
                model,
                &format!("blk.{}.ffn_up.weight", layer),
                n_embd,
                n_ff,
            );
            // FFN down: (n_embd, n_ff) → transpose to (n_ff, n_embd)
            Self::cache_transposed(
                &mut cache,
                model,
                &format!("blk.{}.ffn_down.weight", layer),
                n_ff,
                n_embd,
            );

            // Norms — stored as-is (small, 1-D)
            Self::cache_raw(
                &mut cache,
                model,
                &format!("blk.{}.attn_norm.weight", layer),
            );
            Self::cache_raw(&mut cache, model, &format!("blk.{}.ffn_norm.weight", layer));
        }

        // Output weight: (n_vocab, n_embd) — stored as-is for gemv
        Self::cache_raw(&mut cache, model, "output.weight");
        Self::cache_raw(&mut cache, model, "output_norm.weight");
        // Embeddings stored raw (lookup-table access pattern)
        Self::cache_raw(&mut cache, model, "token_embd.weight");

        let total_floats: usize = cache.values().map(|w| w.data.len()).sum();
        let memory_mb = (total_floats * 4) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "Weight cache ready: {} tensors, {:.1} MB",
            cache.len(),
            memory_mb
        );

        Ok(())
    }

    /// Cache tensor in transposed layout (K, N): weight is (out, in), store as (in, out)
    fn cache_transposed(
        cache: &mut HashMap<String, CachedWeight>,
        model: &crate::loader::Model,
        name: &str,
        in_dim: usize,  // K
        out_dim: usize, // N
    ) {
        let tensor = match model.get_tensor_blocking(name) {
            Some(t) => t,
            None => return,
        };
        let raw = match tensor.as_f32_slice() {
            Ok(d) => d.to_vec(),
            Err(_) => return,
        };
        if raw.len() != out_dim * in_dim {
            // Store as-is if size mismatch (will fall back to fallback path)
            cache.insert(
                name.to_string(),
                CachedWeight {
                    data: raw,
                    rows: 0,
                    cols: 0,
                },
            );
            return;
        }
        // Transpose: W is (out_dim, in_dim), we want (in_dim, out_dim)
        let mut transposed = vec![0.0f32; in_dim * out_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                transposed[i * out_dim + o] = raw[o * in_dim + i];
            }
        }
        cache.insert(
            name.to_string(),
            CachedWeight {
                data: transposed,
                rows: in_dim,
                cols: out_dim,
            },
        );
    }

    /// Cache tensor raw (no transpose)
    fn cache_raw(
        cache: &mut HashMap<String, CachedWeight>,
        model: &crate::loader::Model,
        name: &str,
    ) {
        let tensor = match model.get_tensor_blocking(name) {
            Some(t) => t,
            None => return,
        };
        let data = match tensor.as_f32_slice() {
            Ok(d) => d.to_vec(),
            Err(_) => return,
        };
        cache.insert(
            name.to_string(),
            CachedWeight {
                data,
                rows: 0,
                cols: 0,
            },
        );
    }

    /// Get pre-transposed weight for direct BLAS A*B call
    /// Returns (data, in_dim, out_dim) where data is (in_dim, out_dim)
    pub fn get_proj(&self, name: &str) -> Option<(Vec<f32>, usize, usize)> {
        let cache = self.cache.read().ok()?;
        cache.get(name).and_then(|w| {
            if w.rows > 0 {
                Some((w.data.clone(), w.rows, w.cols))
            } else {
                None
            }
        })
    }

    /// Get raw cached weight (for norms, embeddings, output.weight)
    pub fn get_raw(&self, name: &str) -> Option<Vec<f32>> {
        let cache = self.cache.read().ok()?;
        cache.get(name).map(|w| w.data.clone())
    }
}

impl Default for WeightCache {
    fn default() -> Self {
        Self::new()
    }
}
