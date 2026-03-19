//! Prompt processing optimization
//!
//! Optimized encoding for prompt tokens with:
//! - Parallel processing for batch prompts
//! - Cached prompt results
//! - Efficient KV cache population

use crate::error::{Error, Result};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

/// Prompt processing cache
pub struct PromptCache {
    /// Cache of processed prompts
    cache: Arc<RwLock<std::collections::HashMap<String, CachedPrompt>>>,
    /// Maximum cache size
    max_size: usize,
}

/// Cached prompt data
#[derive(Clone)]
pub struct CachedPrompt {
    /// Token IDs
    pub tokens: Vec<u32>,
    /// Processed timestamp
    pub timestamp: std::time::Instant,
}

impl PromptCache {
    /// Create new prompt cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
            max_size,
        }
    }

    /// Get cached prompt if available
    pub async fn get(&self, key: &str) -> Option<Vec<u32>> {
        let mut cache = self.cache.write().await;
        cache.get_mut(key).map(|cached| {
            cached.timestamp = std::time::Instant::now();
            cached.tokens.clone()
        })
    }

    /// Insert prompt into cache
    pub async fn insert(&self, key: String, tokens: Vec<u32>) {
        let mut cache = self.cache.write().await;

        // Evict oldest if cache is full
        if cache.len() >= self.max_size {
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(
            key,
            CachedPrompt {
                tokens,
                timestamp: std::time::Instant::now(),
            },
        );
    }

    /// Clear cache
    pub async fn clear(&self) {
        self.cache.write().await.clear();
    }

    /// Get cache size
    pub async fn size(&self) -> usize {
        self.cache.read().await.len()
    }
}

/// Parallel prompt processor
pub struct ParallelPromptProcessor {
    /// Prompt cache
    cache: PromptCache,
    /// Number of parallel workers
    workers: usize,
}

impl ParallelPromptProcessor {
    /// Create new parallel prompt processor
    pub fn new(cache_size: usize, workers: usize) -> Self {
        Self {
            cache: PromptCache::new(cache_size),
            workers,
        }
    }

    /// Process multiple prompts in parallel
    pub async fn process_batch(
        &self,
        prompts: Vec<String>,
        encode_fn: impl Fn(&str) -> Result<Vec<u32>> + Send + Sync + Clone + 'static,
    ) -> Result<Vec<Vec<u32>>> {
        use tokio::task::{spawn_blocking, JoinSet};

        let worker_limit = self.workers.max(1);
        let semaphore = Arc::new(Semaphore::new(worker_limit));
        let mut join_set = JoinSet::new();
        let total = prompts.len();

        for (index, prompt) in prompts.into_iter().enumerate() {
            let permit = semaphore.clone();
            let encode = encode_fn.clone();
            join_set.spawn(async move {
                let _permit = permit
                    .acquire_owned()
                    .await
                    .map_err(|e| Error::tensor(format!("Prompt worker exhausted: {}", e)))?;

                let tokens = spawn_blocking(move || encode(&prompt))
                    .await
                    .map_err(|e| Error::tensor(format!("Prompt worker failed: {}", e)))??;

                Ok::<(usize, Vec<u32>), Error>((index, tokens))
            });
        }

        let mut results: Vec<Option<Vec<u32>>> = vec![None; total];

        while let Some(result) = join_set.join_next().await {
            let (index, tokens) =
                result.map_err(|e| Error::tensor(format!("Task join error: {}", e)))??;
            if let Some(slot) = results.get_mut(index) {
                *slot = Some(tokens);
            }
        }

        results
            .into_iter()
            .map(|maybe_tokens| {
                maybe_tokens.ok_or_else(|| Error::tensor("Missing prompt batch result"))
            })
            .collect()
    }

    /// Process single prompt with caching
    pub async fn process_with_cache(
        &self,
        prompt: &str,
        cache_key: String,
        encode_fn: impl Fn(&str) -> Result<Vec<u32>>,
    ) -> Result<Vec<u32>> {
        // Check cache first
        if let Some(tokens) = self.cache.get(cache_key.as_str()).await {
            return Ok(tokens);
        }

        // Encode prompt
        let tokens = encode_fn(prompt)?;

        // Insert into cache
        self.cache.insert(cache_key, tokens.clone()).await;

        Ok(tokens)
    }

    /// Process a batch of prompts with per-item cache keys.
    ///
    /// Cached prompts are returned immediately; misses are encoded in parallel
    /// and inserted back into the cache after completion.
    pub async fn process_batch_with_cache(
        &self,
        prompts: Vec<(String, String)>,
        encode_fn: impl Fn(&str) -> Result<Vec<u32>> + Send + Sync + Clone + 'static,
    ) -> Result<Vec<Vec<u32>>> {
        let mut ordered: Vec<Option<Vec<u32>>> = Vec::with_capacity(prompts.len());
        let mut misses: Vec<(usize, String, String)> = Vec::new();

        for (index, (prompt, cache_key)) in prompts.into_iter().enumerate() {
            if let Some(tokens) = self.cache.get(cache_key.as_str()).await {
                ordered.push(Some(tokens));
            } else {
                ordered.push(None);
                misses.push((index, prompt, cache_key));
            }
        }

        if !misses.is_empty() {
            let encoded = self
                .process_batch(
                    misses.iter().map(|(_, prompt, _)| prompt.clone()).collect(),
                    encode_fn,
                )
                .await?;

            for ((index, _, cache_key), tokens) in misses.into_iter().zip(encoded.into_iter()) {
                self.cache.insert(cache_key, tokens.clone()).await;
                if let Some(slot) = ordered.get_mut(index) {
                    *slot = Some(tokens);
                }
            }
        }

        ordered
            .into_iter()
            .map(|maybe_tokens| {
                maybe_tokens.ok_or_else(|| Error::tensor("Missing cached prompt result"))
            })
            .collect()
    }

    /// Get cache reference
    pub fn cache(&self) -> &PromptCache {
        &self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prompt_cache() {
        let cache = PromptCache::new(10);

        // Insert prompt
        cache.insert("test".to_string(), vec![1, 2, 3]).await;

        // Get cached prompt
        let tokens = cache.get("test").await;
        assert_eq!(tokens, Some(vec![1, 2, 3]));

        // Cache miss
        let tokens = cache.get("not_found").await;
        assert_eq!(tokens, None);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = PromptCache::new(2);

        cache.insert("a".to_string(), vec![1]).await;
        cache.insert("b".to_string(), vec![2]).await;
        cache.insert("c".to_string(), vec![3]).await;

        // Should have evicted one entry
        assert_eq!(cache.size().await, 2);
    }

    #[tokio::test]
    async fn test_parallel_processor() {
        let processor = ParallelPromptProcessor::new(10, 4);

        let prompts = vec!["hello".to_string(), "world".to_string()];

        let results = processor
            .process_batch(prompts, |s| Ok(vec![s.len() as u32]))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_parallel_processor_preserves_order() {
        let processor = ParallelPromptProcessor::new(10, 2);

        let prompts = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];

        let results = processor
            .process_batch(prompts, |s| Ok(vec![s.len() as u32]))
            .await
            .unwrap();

        assert_eq!(results, vec![vec![1], vec![2], vec![3]]);
    }

    #[tokio::test]
    async fn test_parallel_processor_batch_cache() {
        let processor = ParallelPromptProcessor::new(10, 2);

        let prompts = vec![
            ("alpha".to_string(), "key-a".to_string()),
            ("beta".to_string(), "key-b".to_string()),
        ];

        let first = processor
            .process_batch_with_cache(prompts.clone(), |s| Ok(vec![s.len() as u32]))
            .await
            .unwrap();
        assert_eq!(first, vec![vec![5], vec![4]]);

        let second = processor
            .process_batch_with_cache(prompts, |_| Ok(vec![99]))
            .await
            .unwrap();
        assert_eq!(second, vec![vec![5], vec![4]]);
    }
}
