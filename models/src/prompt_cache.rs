//! Prompt caching for KV cache
//!
//! Implements prefix caching to avoid recomputing KV states for repeated prompts.
//! This is especially useful for:
//! - Repeated system prompts
//! - RAG preambles
//! - Chat templates
//! - Few-shot examples

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::kv_cache::AdvancedKVCache;
use crate::context::ModelContext;
use core::error::{Error, Result};

/// Cache key for prompt KV state
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct PromptCacheKey {
    /// Hash of the prompt tokens
    pub prompt_hash: u64,
    /// Model identifier
    pub model_id: String,
    /// Context length
    pub context_length: usize,
}

/// Cached prompt KV state
#[derive(Debug, Clone)]
pub struct CachedPromptState {
    /// The cache key
    pub key: PromptCacheKey,
    /// KV cache data
    pub kv_data: Vec<u8>,
    /// Number of tokens in cache
    pub token_count: usize,
    /// Timestamp when cached
    pub timestamp: std::time::Instant,
    /// Access count for LRU eviction
    pub access_count: usize,
}

/// Prompt cache for KV states
pub struct PromptCache {
    /// Cache storage
    cache: Arc<RwLock<HashMap<PromptCacheKey, CachedPromptState>>>,
    /// Maximum cache size in bytes
    max_size_bytes: usize,
    /// Current cache size in bytes
    current_size_bytes: Arc<RwLock<usize>>,
}

impl PromptCache {
    /// Create a new prompt cache
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size_bytes: max_size_mb * 1024 * 1024,
            current_size_bytes: Arc::new(RwLock::new(0)),
        }
    }

    /// Get cached KV state for a prompt
    pub fn get(&self, key: &PromptCacheKey) -> Option<CachedPromptState> {
        let mut cache = self.cache.write().unwrap();
        if let Some(mut state) = cache.get_mut(key) {
            state.access_count += 1;
            Some(state.clone())
        } else {
            None
        }
    }

    /// Store KV state for a prompt
    pub fn put(&self, key: PromptCacheKey, kv_data: Vec<u8>, token_count: usize) -> Result<()> {
        let size = kv_data.len();

        // Check if we need to evict
        {
            let current_size = *self.current_size_bytes.read().unwrap();
            if current_size + size > self.max_size_bytes {
                self.evict(size)?;
            }
        }

        let state = CachedPromptState {
            key: key.clone(),
            kv_data,
            token_count,
            timestamp: std::time::Instant::now(),
            access_count: 1,
        };

        let mut cache = self.cache.write().unwrap();
        let mut current_size = self.current_size_bytes.write().unwrap();

        cache.insert(key, state);
        *current_size += size;

        Ok(())
    }

    /// Evict least recently used entries
    fn evict(&self, required_bytes: usize) -> Result<()> {
        let mut cache = self.cache.write().unwrap();
        let mut current_size = self.current_size_bytes.write().unwrap();

        let mut to_remove = Vec::new();
        let mut freed = 0;

        // Sort by access count and timestamp (LRU)
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by(|a, b| {
            a.1.access_count.cmp(&b.1.access_count)
                .then(a.1.timestamp.cmp(&b.1.timestamp))
        });

        for (key, state) in entries {
            if freed >= required_bytes {
                break;
            }
            to_remove.push(key.clone());
            freed += state.kv_data.len();
        }

        for key in to_remove {
            if let Some(state) = cache.remove(&key) {
                *current_size -= state.kv_data.len();
            }
        }

        Ok(())
    }

    /// Clear all cached prompts
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut current_size = self.current_size_bytes.write().unwrap();
        cache.clear();
        *current_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> PromptCacheStats {
        let cache = self.cache.read().unwrap();
        let current_size = *self.current_size_bytes.read().unwrap();

        PromptCacheStats {
            entries: cache.len(),
            size_bytes: current_size,
            max_size_bytes: self.max_size_bytes,
            utilization: if self.max_size_bytes > 0 {
                current_size as f64 / self.max_size_bytes as f64
            } else {
                0.0
            },
        }
    }
}

/// Prompt cache statistics
#[derive(Debug, Clone)]
pub struct PromptCacheStats {
    pub entries: usize,
    pub size_bytes: usize,
    pub max_size_bytes: usize,
    pub utilization: f64,
}

impl PromptCacheStats {
    pub fn print(&self) {
        println!("\n=== Prompt Cache Stats ===");
        println!("Entries:     {}", self.entries);
        println!("Size:         {:.2} MB", self.size_bytes as f64 / (1024.0 * 1024.0));
        println!("Max size:     {:.2} MB", self.max_size_bytes as f64 / (1024.0 * 1024.0));
        println!("Utilization:  {:.1}%", self.utilization * 100.0);
        println!("===========================\n");
    }
}

/// Compute cache key from prompt tokens
pub fn compute_cache_key(tokens: &[i32], model_id: &str) -> PromptCacheKey {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    model_id.hash(&mut hasher);

    PromptCacheKey {
        prompt_hash: hasher.finish(),
        model_id: model_id.to_string(),
        context_length: tokens.len(),
    }
}

/// Save KV cache state to bytes
pub fn save_kv_state(ctx: &ModelContext) -> Result<Vec<u8>> {
    // TODO: Implement actual KV state serialization
    // For now, return empty vector
    Ok(Vec::new())
}

/// Restore KV cache state from bytes
pub fn restore_kv_state(ctx: &mut ModelContext, data: &[u8]) -> Result<()> {
    // TODO: Implement actual KV state deserialization
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_cache_creation() {
        let cache = PromptCache::new(100); // 100 MB
        assert_eq!(cache.max_size_bytes, 100 * 1024 * 1024);
    }

    #[test]
    fn test_compute_cache_key() {
        let tokens = vec![1, 2, 3, 4, 5];
        let key1 = compute_cache_key(&tokens, "model1");
        let key2 = compute_cache_key(&tokens, "model1");
        let key3 = compute_cache_key(&tokens, "model2");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_prompt_cache_put_get() {
        let cache = PromptCache::new(100);
        let key = PromptCacheKey {
            prompt_hash: 123,
            model_id: "test".to_string(),
            context_length: 10,
        };

        let kv_data = vec![1u8; 1000];
        cache.put(key.clone(), kv_data, 10).unwrap();

        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().token_count, 10);
    }

    #[test]
    fn test_prompt_cache_stats() {
        let cache = PromptCache::new(100);
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.utilization, 0.0);
    }
}
