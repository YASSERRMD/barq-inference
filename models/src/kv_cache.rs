//! Advanced KV cache with defragmentation and efficient memory management
//!
//! This implements an optimized KV cache with:
//! - Ring buffer structure for continuous allocation
//! - Automatic defragmentation to reclaim space
//! - Cache statistics for monitoring
//! - Efficient token management for long contexts

use std::collections::VecDeque;
use barq_core::error::{Error, Result};
use barq_core::tensor::{Tensor, TensorType, TensorData, Shape};

/// KV cache entry for a single token position
#[derive(Debug, Clone)]
struct KVCacheEntry {
    /// Key tensor for this position [num_heads, head_dim]
    k: Tensor,
    /// Value tensor for this position [num_heads, head_dim]
    v: Tensor,
    /// Physical position in the cache (may differ from logical position)
    physical_pos: usize,
}

/// Statistics about the KV cache
#[derive(Debug, Clone, Default)]
pub struct KVCacheStats {
    /// Total number of tokens currently cached
    pub tokens_cached: usize,
    /// Total cache capacity in tokens
    pub capacity: usize,
    /// Number of defragmentation operations performed
    pub defrag_count: usize,
    /// Current memory fragmentation ratio (0.0 = none, 1.0 = fully fragmented)
    pub fragmentation_ratio: f32,
    /// Cache hits (tokens served from cache)
    pub cache_hits: usize,
    /// Cache misses (tokens computed fresh)
    pub cache_misses: usize,
}

impl KVCacheStats {
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

/// Advanced KV cache with automatic defragmentation
pub struct AdvancedKVCache {
    /// Ring buffer of KV entries
    cache: VecDeque<Option<KVCacheEntry>>,
    /// Logical position to physical position mapping
    logical_to_physical: Vec<Option<usize>>,
    /// Physical position to logical position mapping
    physical_to_logical: Vec<Option<usize>>,
    /// Number of KV heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Maximum cache capacity in tokens
    capacity: usize,
    /// Current number of tokens in cache
    len: usize,
    /// Defragmentation threshold (0.0-1.0)
    defrag_threshold: f32,
    /// Statistics
    stats: KVCacheStats,
}

impl AdvancedKVCache {
    /// Create a new KV cache
    pub fn new(num_heads: usize, head_dim: usize, capacity: usize) -> Self {
        let mut cache = VecDeque::with_capacity(capacity);
        for _ in 0..capacity {
            cache.push_back(None);
        }

        Self {
            cache,
            logical_to_physical: vec![None; capacity],
            physical_to_logical: vec![None; capacity],
            num_heads,
            head_dim,
            capacity,
            len: 0,
            defrag_threshold: 0.3, // Defrag when 30% fragmented
            stats: KVCacheStats {
                capacity,
                ..Default::default()
            },
        }
    }

    /// Set defragmentation threshold
    pub fn with_defrag_threshold(mut self, threshold: f32) -> Self {
        self.defrag_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Insert KV pair for a token position
    pub fn insert(&mut self, pos: usize, k: Tensor, v: Tensor) -> Result<()> {
        // Validate tensor shapes
        let k_shape = k.shape();
        let v_shape = v.shape();

        if k_shape.dims().len() != 2 || v_shape.dims().len() != 2 {
            return Err(Error::tensor("K and V must be 2D tensors [num_heads, head_dim]"));
        }

        if k_shape.dims()[0] != self.num_heads || k_shape.dims()[1] != self.head_dim {
            return Err(Error::tensor("K tensor shape mismatch"));
        }

        if v_shape.dims()[0] != self.num_heads || v_shape.dims()[1] != self.head_dim {
            return Err(Error::tensor("V tensor shape mismatch"));
        }

        // Check if position already exists
        if pos < self.logical_to_physical.len() && self.logical_to_physical[pos].is_some() {
            // Update existing entry
            let physical_pos = self.logical_to_physical[pos].unwrap();
            if let Some(entry) = self.cache.get_mut(physical_pos) {
                *entry = Some(KVCacheEntry {
                    k,
                    v,
                    physical_pos,
                });
            }
        } else {
            // Insert new entry
            if self.len >= self.capacity {
                // Need to evict or defrag
                self.defragment()?;
            }

            // Find empty slot
            let physical_pos = self.find_free_slot().ok_or_else(|| {
                Error::tensor("KV cache is full, cannot insert")
            })?;

            self.cache[physical_pos] = Some(KVCacheEntry {
                k,
                v,
                physical_pos,
            });

            // Update mappings
            if pos >= self.logical_to_physical.len() {
                self.logical_to_physical.resize(pos + 1, None);
            }
            self.logical_to_physical[pos] = Some(physical_pos);
            self.physical_to_logical[physical_pos] = Some(pos);
            self.len += 1;
        }

        self.update_stats();
        Ok(())
    }

    /// Get KV pair for a token position
    pub fn get(&mut self, pos: usize) -> Result<Option<(Tensor, Tensor)>> {
        if pos >= self.logical_to_physical.len() {
            self.stats.cache_misses += 1;
            return Ok(None);
        }

        let physical_pos = match self.logical_to_physical[pos] {
            Some(pp) => pp,
            None => {
                self.stats.cache_misses += 1;
                return Ok(None);
            }
        };

        match &self.cache[physical_pos] {
            Some(entry) => {
                self.stats.cache_hits += 1;
                Ok(Some((entry.k.clone(), entry.v.clone())))
            }
            None => {
                self.stats.cache_misses += 1;
                Ok(None)
            }
        }
    }

    /// Get all cached K tensors up to a position
    pub fn get_k_prefix(&mut self, end_pos: usize) -> Result<Vec<Tensor>> {
        let mut result = Vec::new();

        for pos in 0..end_pos {
            if let Some((k, _)) = self.get(pos)? {
                result.push(k);
            }
        }

        Ok(result)
    }

    /// Get all cached V tensors up to a position
    pub fn get_v_prefix(&mut self, end_pos: usize) -> Result<Vec<Tensor>> {
        let mut result = Vec::new();

        for pos in 0..end_pos {
            if let Some((_, v)) = self.get(pos)? {
                result.push(v);
            }
        }

        Ok(result)
    }

    /// Find a free slot in the cache
    fn find_free_slot(&self) -> Option<usize> {
        for (i, entry) in self.cache.iter().enumerate() {
            if entry.is_none() {
                return Some(i);
            }
        }
        None
    }

    /// Defragment the cache by compacting entries
    pub fn defragment(&mut self) -> Result<()> {
        self.stats.defrag_count += 1;

        // Collect all valid entries in logical order
        let mut entries = Vec::new();
        for logical_pos in 0..self.logical_to_physical.len() {
            if let Some(physical_pos) = self.logical_to_physical[logical_pos] {
                if let Some(entry) = &self.cache[physical_pos] {
                    entries.push((logical_pos, entry.clone()));
                }
            }
        }

        // Clear cache
        for entry in self.cache.iter_mut() {
            *entry = None;
        }
        self.logical_to_physical.clear();
        self.physical_to_logical = vec![None; self.capacity];

        // Rebuild cache in compact form
        for (new_physical_pos, (logical_pos, entry)) in entries.iter().enumerate() {
            let mut new_entry = entry.clone();
            new_entry.physical_pos = new_physical_pos;

            self.cache[new_physical_pos] = Some(new_entry);
            self.logical_to_physical.push(Some(new_physical_pos));
            self.physical_to_logical[new_physical_pos] = Some(*logical_pos);
        }

        self.len = entries.len();
        self.update_stats();

        Ok(())
    }

    /// Update cache statistics
    fn update_stats(&mut self) {
        self.stats.tokens_cached = self.len;

        // Calculate fragmentation ratio
        let mut gaps = 0;
        let mut last_physical = None;

        for logical_pos in 0..self.logical_to_physical.len() {
            if let Some(physical_pos) = self.logical_to_physical[logical_pos] {
                if let Some(last) = last_physical {
                    if physical_pos != last + 1 {
                        gaps += 1;
                    }
                }
                last_physical = Some(physical_pos);
            }
        }

        self.stats.fragmentation_ratio = if self.len > 1 {
            gaps as f32 / (self.len - 1) as f32
        } else {
            0.0
        };
    }

    /// Check if defragmentation is needed
    pub fn needs_defrag(&self) -> bool {
        self.stats.fragmentation_ratio > self.defrag_threshold
    }

    /// Clear the entire cache
    pub fn clear(&mut self) {
        for entry in self.cache.iter_mut() {
            *entry = None;
        }
        self.logical_to_physical.clear();
        self.physical_to_logical = vec![None; self.capacity];
        self.len = 0;
        self.update_stats();
    }

    /// Get cache statistics
    pub fn stats(&self) -> &KVCacheStats {
        &self.stats
    }

    /// Get current cache length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let cache = AdvancedKVCache::new(32, 128, 1024);
        assert_eq!(cache.num_heads, 32);
        assert_eq!(cache.head_dim, 128);
        assert_eq!(cache.capacity(), 1024);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_insert() {
        let mut cache = AdvancedKVCache::new(2, 4, 10);

        let k_shape = Shape::new(&[2, 4]);
        let v_shape = Shape::new(&[2, 4]);

        let k = Tensor::new(None, TensorType::F32, k_shape, TensorData::F32(vec![1.0f32; 8])).unwrap();
        let v = Tensor::new(None, TensorType::F32, v_shape, TensorData::F32(vec![2.0f32; 8])).unwrap();

        cache.insert(0, k, v).unwrap();

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_kv_cache_get() {
        let mut cache = AdvancedKVCache::new(2, 4, 10);

        let k_shape = Shape::new(&[2, 4]);
        let v_shape = Shape::new(&[2, 4]);

        let k = Tensor::new(None, TensorType::F32, k_shape, TensorData::F32(vec![1.0f32; 8])).unwrap();
        let v = Tensor::new(None, TensorType::F32, v_shape, TensorData::F32(vec![2.0f32; 8])).unwrap();

        cache.insert(0, k.clone(), v.clone()).unwrap();

        let result = cache.get(0).unwrap();
        assert!(result.is_some());

        let (retrieved_k, retrieved_v) = result.unwrap();
        assert_eq!(retrieved_k.shape().dims, k.shape().dims);
        assert_eq!(retrieved_v.shape().dims, v.shape().dims);
    }

    #[test]
    fn test_kv_cache_stats() {
        let mut cache = AdvancedKVCache::new(2, 4, 10);

        let k_shape = Shape::new(&[2, 4]);
        let v_shape = Shape::new(&[2, 4]);

        let k = Tensor::new(None, TensorType::F32, k_shape, TensorData::F32(vec![1.0f32; 8])).unwrap();
        let v = Tensor::new(None, TensorType::F32, v_shape, TensorData::F32(vec![2.0f32; 8])).unwrap();

        cache.insert(0, k, v).unwrap();
        cache.get(0).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.tokens_cached, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.hit_rate(), 1.0);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = AdvancedKVCache::new(2, 4, 10);

        let k_shape = Shape::new(&[2, 4]);
        let v_shape = Shape::new(&[2, 4]);

        let k = Tensor::new(None, TensorType::F32, k_shape, TensorData::F32(vec![1.0f32; 8])).unwrap();
        let v = Tensor::new(None, TensorType::F32, v_shape, TensorData::F32(vec![2.0f32; 8])).unwrap();

        cache.insert(0, k, v).unwrap();
        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
