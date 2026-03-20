//! Weight caching system to avoid repeated dequantization
//!
//! Stores dequantized weights in memory to eliminate the overhead of
//! repeated GGUF dequantization on every forward pass.
//!
//! Linear tensors are kept in transposed row-major layout for direct BLAS
//! calls. Embeddings and norms stay in their raw GGUF layout.

use barq_core::error::{Error, Result};
use barq_core::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;

/// Cached weight — stored either transposed for BLAS or raw for lookup tensors.
pub struct CachedWeight {
    pub data: CachedWeightData,
    /// rows (in_dim) for transposed projection weights
    pub rows: usize,
    /// cols (out_dim) for transposed projection weights
    pub cols: usize,
}

/// Cached weight payload.
pub enum CachedWeightData {
    /// Raw tensor shared from the model cache.
    Raw(Arc<Tensor>),
    /// Dequantized and transposed f32 data used by BLAS.
    Transposed(Arc<[f32]>),
}

/// Weight cache for storing pre-dequantized, pre-transposed weights
pub struct WeightCache {
    /// Cached tensors indexed by tensor name
    cache: RwLock<HashMap<String, CachedWeight>>,
}

impl WeightCache {
    /// Create a new weight cache
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize the cache with all required weights in the layout used by
    /// the forward pass.
    pub fn initialize(&self, model: &crate::loader::Model, n_layer: usize) -> Result<()> {
        eprintln!("Initializing weight cache (dequantize + transpose projections)...");

        let n_embd = model.hparams().n_embd as usize;
        let n_head = model.hparams().n_head as usize;
        let n_head_kv = model.hparams().n_head_kv as usize;
        let n_ff = model.hparams().n_ff as usize;
        let head_dim = n_embd / n_head;

        #[derive(Debug)]
        enum CacheSpec {
            Transposed {
                name: String,
                in_dim: usize,
                out_dim: usize,
            },
            Raw {
                name: String,
            },
        }

        let mut specs = Vec::new();
        for layer in 0..n_layer {
            specs.push(CacheSpec::Transposed {
                name: format!("blk.{}.attn_q.weight", layer),
                in_dim: n_embd,
                out_dim: n_head * head_dim,
            });
            specs.push(CacheSpec::Transposed {
                name: format!("blk.{}.attn_k.weight", layer),
                in_dim: n_embd,
                out_dim: n_head_kv * head_dim,
            });
            specs.push(CacheSpec::Transposed {
                name: format!("blk.{}.attn_v.weight", layer),
                in_dim: n_embd,
                out_dim: n_head_kv * head_dim,
            });
            specs.push(CacheSpec::Transposed {
                name: format!("blk.{}.attn_output.weight", layer),
                in_dim: n_embd,
                out_dim: n_embd,
            });
            specs.push(CacheSpec::Transposed {
                name: format!("blk.{}.ffn_gate.weight", layer),
                in_dim: n_embd,
                out_dim: n_ff,
            });
            specs.push(CacheSpec::Transposed {
                name: format!("blk.{}.ffn_up.weight", layer),
                in_dim: n_embd,
                out_dim: n_ff,
            });
            specs.push(CacheSpec::Transposed {
                name: format!("blk.{}.ffn_down.weight", layer),
                in_dim: n_ff,
                out_dim: n_embd,
            });

            specs.push(CacheSpec::Raw {
                name: format!("blk.{}.attn_norm.weight", layer),
            });
            specs.push(CacheSpec::Raw {
                name: format!("blk.{}.attn_q.bias", layer),
            });
            specs.push(CacheSpec::Raw {
                name: format!("blk.{}.attn_k.bias", layer),
            });
            specs.push(CacheSpec::Raw {
                name: format!("blk.{}.attn_v.bias", layer),
            });
            specs.push(CacheSpec::Raw {
                name: format!("blk.{}.ffn_norm.weight", layer),
            });
        }

        specs.push(CacheSpec::Raw {
            name: "output.weight".to_string(),
        });
        specs.push(CacheSpec::Raw {
            name: "output.bias".to_string(),
        });
        specs.push(CacheSpec::Raw {
            name: "output_norm.weight".to_string(),
        });
        specs.push(CacheSpec::Raw {
            name: "output_norm.bias".to_string(),
        });
        specs.push(CacheSpec::Raw {
            name: "token_embd.weight".to_string(),
        });

        let entries: Vec<(String, CachedWeight)> = specs
            .par_iter()
            .filter_map(|spec| match spec {
                CacheSpec::Transposed {
                    name,
                    in_dim,
                    out_dim,
                } => Self::build_transposed_entry(model, name, *in_dim, *out_dim),
                CacheSpec::Raw { name } => Self::build_raw_entry(model, name),
            })
            .collect();

        let mut cache = self
            .cache
            .write()
            .map_err(|e| Error::backend(format!("Cache write lock failed: {}", e)))?;

        for (name, entry) in entries {
            cache.insert(name, entry);
        }

        let total_floats: usize = cache.values().map(|w| w.len()).sum();
        let memory_mb = (total_floats * 4) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "Weight cache ready: {} tensors, {:.1} MB",
            cache.len(),
            memory_mb
        );

        Ok(())
    }

    fn build_transposed_entry(
        model: &crate::loader::Model,
        name: &str,
        in_dim: usize,  // K
        out_dim: usize, // N
    ) -> Option<(String, CachedWeight)> {
        let tensor = match model.get_tensor_blocking(name) {
            Some(t) => t,
            None => return None,
        };
        let raw = match tensor.as_f32_slice() {
            Ok(d) => d.to_vec(),
            Err(_) => return None,
        };
        if raw.len() != out_dim * in_dim {
            return Some((
                name.to_string(),
                CachedWeight {
                    data: CachedWeightData::Transposed(Arc::from(raw)),
                    rows: 0,
                    cols: 0,
                },
            ));
        }
        let mut transposed = vec![0.0f32; in_dim * out_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                transposed[i * out_dim + o] = raw[o * in_dim + i];
            }
        }
        Some((
            name.to_string(),
            CachedWeight {
                data: CachedWeightData::Transposed(Arc::from(transposed)),
                rows: in_dim,
                cols: out_dim,
            },
        ))
    }

    fn build_raw_entry(
        model: &crate::loader::Model,
        name: &str,
    ) -> Option<(String, CachedWeight)> {
        let tensor = match model.get_tensor_blocking(name) {
            Some(t) => t,
            None => return None,
        };
        Some((
            name.to_string(),
            CachedWeight {
                data: CachedWeightData::Raw(Arc::new(tensor)),
                rows: 0,
                cols: 0,
            },
        ))
    }

    /// Get cached projection weight for direct BLAS A*B call.
    /// Returns (data, in_dim, out_dim) where data is (in_dim, out_dim).
    pub fn get_proj(&self, name: &str) -> Option<(Arc<[f32]>, usize, usize)> {
        let cache = self.cache.read().ok()?;
        cache.get(name).and_then(|w| {
            if w.rows > 0 {
                match &w.data {
                    CachedWeightData::Transposed(data) => Some((Arc::clone(data), w.rows, w.cols)),
                    CachedWeightData::Raw(_) => None,
                }
            } else {
                None
            }
        })
    }

    /// Get raw cached weight (for norms, embeddings, output.weight)
    pub fn get_raw(&self, name: &str) -> Option<Arc<Tensor>> {
        let cache = self.cache.read().ok()?;
        cache.get(name).and_then(|w| match &w.data {
            CachedWeightData::Raw(data) => Some(Arc::clone(data)),
            CachedWeightData::Transposed(_) => None,
        })
    }
}

impl CachedWeight {
    fn len(&self) -> usize {
        match &self.data {
            CachedWeightData::Raw(_) => 0,
            CachedWeightData::Transposed(data) => data.len(),
        }
    }
}

impl Default for WeightCache {
    fn default() -> Self {
        Self::new()
    }
}
