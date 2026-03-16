//! KV cache statistics and monitoring
//!
//! Provides real-time metrics for KV cache performance monitoring

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// KV cache statistics
#[derive(Debug)]
pub struct KVCacheStats {
    /// Total cache hits
    pub cache_hits: AtomicU64,
    /// Total cache misses
    pub cache_misses: AtomicU64,
    /// Total tokens cached
    pub total_tokens_cached: AtomicU64,
    /// Current cache size in tokens
    pub current_cache_size: AtomicUsize,
    /// Peak cache size in tokens
    pub peak_cache_size: AtomicUsize,
    /// Number of defragmentation operations
    pub defrag_count: AtomicU64,
    /// Number of cache evictions
    pub eviction_count: AtomicU64,
}

impl KVCacheStats {
    pub fn new() -> Self {
        Self {
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_tokens_cached: AtomicU64::new(0),
            current_cache_size: AtomicUsize::new(0),
            peak_cache_size: AtomicUsize::new(0),
            defrag_count: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
        }
    }

    /// Record a cache hit
    pub fn record_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record tokens added to cache
    pub fn record_tokens_cached(&self, count: usize) {
        self.total_tokens_cached.fetch_add(count as u64, Ordering::Relaxed);
        let old_size = self.current_cache_size.fetch_add(count, Ordering::Relaxed);

        // Update peak
        let new_size = old_size + count;
        let mut peak = self.peak_cache_size.load(Ordering::Relaxed);
        while new_size > peak {
            match self.peak_cache_size.compare_exchange_weak(
                peak,
                new_size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Record defragmentation operation
    pub fn record_defrag(&self) {
        self.defrag_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache eviction
    pub fn record_eviction(&self) {
        self.evacuation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Print statistics
    pub fn print(&self) {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens_cached.load(Ordering::Relaxed);
        let current_size = self.current_cache_size.load(Ordering::Relaxed);
        let peak_size = self.peak_cache_size.load(Ordering::Relaxed);
        let defrags = self.defrag_count.load(Ordering::Relaxed);
        let evictions = self.evacuation_count.load(Ordering::Relaxed);

        println!("\n=== KV Cache Statistics ===");
        println!("Cache hits:        {}", hits);
        println!("Cache misses:      {}", misses);
        println!("Hit rate:          {:.2}%", self.hit_rate() * 100.0);
        println!("Tokens cached:     {}", total_tokens);
        println!("Current size:      {} tokens", current_size);
        println!("Peak size:         {} tokens", peak_size);
        println!("Defragmentations:  {}", defrags);
        println!("Evictions:         {}", evictions);
        println!("==============================\n");
    }
}

impl Default for KVCacheStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Global KV cache monitor for tracking multiple caches
pub struct KVCacheMonitor {
    stats: Arc<KVCacheStats>,
}

impl KVCacheMonitor {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(KVCacheStats::new()),
        }
    }

    pub fn stats(&self) -> &Arc<KVCacheStats> {
        &self.stats
    }

    /// Export statistics as JSON for metrics endpoint
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&*self.stats)
    }
}

/// Serializable stats for JSON export
#[derive(serde::Serialize)]
pub struct SerializableStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f64,
    pub total_tokens_cached: u64,
    pub current_cache_size: usize,
    pub peak_cache_size: usize,
    pub defrag_count: u64,
    pub eviction_count: u64,
}

impl From<&KVCacheStats> for SerializableStats {
    fn from(stats: &KVCacheStats) -> Self {
        Self {
            cache_hits: stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: stats.cache_misses.load(Ordering::Relaxed),
            hit_rate: stats.hit_rate(),
            total_tokens_cached: stats.total_tokens_cached.load(Ordering::Relaxed),
            current_cache_size: stats.current_cache_size.load(Ordering::Relaxed),
            peak_cache_size: stats.peak_cache_size.load(Ordering::Relaxed),
            defrag_count: stats.defrag_count.load(Ordering::Relaxed),
            eviction_count: stats.evacuation_count.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_stats_creation() {
        let stats = KVCacheStats::new();
        assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 0);
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_kv_cache_stats_hit_rate() {
        let stats = KVCacheStats::new();

        stats.record_hit();
        stats.record_hit();
        stats.record_miss();

        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_kv_cache_stats_tokens() {
        let stats = KVCacheStats::new();

        stats.record_tokens_cached(100);
        stats.record_tokens_cached(50);

        assert_eq!(stats.total_tokens_cached.load(Ordering::Relaxed), 150);
        assert_eq!(stats.current_cache_size.load(Ordering::Relaxed), 150);
        assert_eq!(stats.peak_cache_size.load(Ordering::Relaxed), 150);
    }

    #[test]
    fn test_kv_cache_stats_peak() {
        let stats = KVCacheStats::new();

        stats.record_tokens_cached(100);
        assert_eq!(stats.peak_cache_size.load(Ordering::Relaxed), 100);

        stats.record_tokens_cached(50);
        assert_eq!(stats.peak_cache_size.load(Ordering::Relaxed), 150);
    }

    #[test]
    fn test_kv_cache_monitor() {
        let monitor = KVCacheMonitor::new();
        monitor.stats().record_hit();
        monitor.stats().record_miss();

        assert!((monitor.stats().hit_rate() - 0.5).abs() < 0.01);
    }
}
