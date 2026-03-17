//! Prompt cache using a Radix Tree for optimal prefix matching.
//!
//! Enables O(k) prefix lookup where k = matched token length, allowing
//! reuse of previously computed KV cache pages across requests that
//! share a common prompt prefix (e.g., system prompts, few-shot examples).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use barq_core::error::Result;

/// A node in the Radix Tree
#[derive(Debug)]
struct RadixNode {
    /// Token sequence stored at this edge from parent to this node
    tokens: Vec<i32>,
    /// KV cache page IDs covering these tokens
    pages: Vec<usize>,
    /// Child nodes keyed by their first token
    children: HashMap<i32, Box<RadixNode>>,
    /// Last access time (for LRU eviction)
    last_accessed: Instant,
    /// Access count
    access_count: usize,
}

impl RadixNode {
    fn new(tokens: Vec<i32>, pages: Vec<usize>) -> Self {
        Self {
            tokens,
            pages,
            children: HashMap::new(),
            last_accessed: Instant::now(),
            access_count: 1,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Result of a prefix match lookup
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// Number of tokens matched from the query
    pub matched_tokens: usize,
    /// KV cache pages that can be reused
    pub pages: Vec<usize>,
}

/// Prompt Cache backed by a Radix Tree.
///
/// Thread-safe via `RwLock`. Supports:
/// - O(k) prefix lookup
/// - Incremental insertion of new prefix segments  
/// - LRU-based eviction when over capacity
pub struct PromptCache {
    root: Arc<RwLock<RadixNode>>,
    /// Maximum number of total pages to hold across all cached prefixes
    max_pages: usize,
    /// Current total page count
    total_pages: Arc<RwLock<usize>>,
}

impl PromptCache {
    /// Create a new prompt cache with the given page capacity
    pub fn new(max_pages: usize) -> Self {
        Self {
            root: Arc::new(RwLock::new(RadixNode::new(vec![], vec![]))),
            max_pages,
            total_pages: Arc::new(RwLock::new(0)),
        }
    }

    /// Look up the longest matching prefix of `tokens` in the cache.
    /// Returns the matched length and reusable pages.
    pub async fn match_prefix(&self, tokens: &[i32]) -> PrefixMatch {
        let mut node = self.root.write().await;
        let mut matched = 0;
        let mut matched_pages = Vec::new();

        Self::match_recursive(&mut node, tokens, &mut matched, &mut matched_pages);

        PrefixMatch {
            matched_tokens: matched,
            pages: matched_pages,
        }
    }

    fn match_recursive(
        node: &mut RadixNode,
        tokens: &[i32],
        matched: &mut usize,
        pages: &mut Vec<usize>,
    ) {
        if tokens.is_empty() {
            return;
        }

        let first = tokens[0];
        if let Some(child) = node.children.get_mut(&first) {
            // Find common prefix length between `tokens` and this child's edge
            let common = tokens
                .iter()
                .zip(child.tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common == child.tokens.len() {
                // Full edge match — consume and recurse
                child.touch();
                *matched += common;
                pages.extend_from_slice(&child.pages);
                let remaining = &tokens[common..];
                Self::match_recursive(child, remaining, matched, pages);
            } else {
                // Partial edge match — stop here
                *matched += common;
                pages.extend_from_slice(&child.pages[..child.pages.len().min(common)]);
            }
        }
    }

    /// Insert a token sequence with its associated KV cache pages into the tree.
    /// Splits existing edges as needed (true radix behaviour).
    pub async fn insert(&self, tokens: &[i32], pages: &[usize]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        let n_pages = pages.len();

        // Check if we need to evict before inserting (read total first, then drop the guard)
        let need_evict = {
            let total = *self.total_pages.read().await;
            total + n_pages > self.max_pages
        };

        if need_evict {
            // Evict under root write-lock only — no re-read of total_pages here
            let mut root = self.root.write().await;
            let mut freed = 0usize;
            Self::evict_recursive(&mut root, n_pages, &mut freed);
            let mut tp = self.total_pages.write().await;
            *tp = tp.saturating_sub(freed);
        }

        {
            let mut root = self.root.write().await;
            Self::insert_recursive(&mut root, tokens, pages);
        }

        *self.total_pages.write().await += n_pages;
        Ok(())
    }

    fn insert_recursive(node: &mut RadixNode, tokens: &[i32], pages: &[usize]) {
        if tokens.is_empty() {
            return;
        }

        let first = tokens[0];

        if let Some(child) = node.children.get_mut(&first) {
            let common = tokens
                .iter()
                .zip(child.tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common == child.tokens.len() {
                // Full edge consumed — recurse deeper
                child.touch();
                Self::insert_recursive(child, &tokens[common..], &pages[common.min(pages.len())..]);
                return;
            }

            // Split the edge at `common`
            let split_tokens = child.tokens[common..].to_vec();
            let split_pages = if common < child.pages.len() {
                child.pages[common..].to_vec()
            } else {
                vec![]
            };

            // Truncate child to common prefix
            child.tokens.truncate(common);
            child.pages.truncate(common);

            // Move child's subtree under a new split node
            let mut split_child = Box::new(RadixNode::new(split_tokens, split_pages));
            split_child.children = std::mem::take(&mut child.children);

            let split_key = split_child.tokens[0];
            child.children.insert(split_key, split_child);

            // Insert the new diverging node
            let new_tokens = tokens[common..].to_vec();
            let new_pages = if common < pages.len() {
                pages[common..].to_vec()
            } else {
                vec![]
            };

            if !new_tokens.is_empty() {
                let new_key = new_tokens[0];
                child
                    .children
                    .insert(new_key, Box::new(RadixNode::new(new_tokens, new_pages)));
            }
        } else {
            // No matching child — insert directly
            node.children.insert(
                first,
                Box::new(RadixNode::new(tokens.to_vec(), pages.to_vec())),
            );
        }
    }

    fn evict_recursive(node: &mut RadixNode, need: usize, freed: &mut usize) {
        if *freed >= need {
            return;
        }

        // Collect children sorted by last access time (oldest first)
        let mut keys: Vec<i32> = node.children.keys().copied().collect();
        keys.sort_by_key(|k| node.children[k].last_accessed);

        let mut to_remove = Vec::new();
        for key in keys {
            if *freed >= need {
                break;
            }
            let child = node.children.get_mut(&key).unwrap();
            if child.children.is_empty() {
                // Leaf node — evict it
                *freed += child.pages.len();
                to_remove.push(key);
            } else {
                // Recurse into children first
                Self::evict_recursive(child, need, freed);
            }
        }

        for key in to_remove {
            node.children.remove(&key);
        }
    }

    /// Return total number of cached pages currently held
    pub async fn cached_pages(&self) -> usize {
        *self.total_pages.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prefix_match_empty() {
        let cache = PromptCache::new(1000);
        let result = cache.match_prefix(&[1, 2, 3]).await;
        assert_eq!(result.matched_tokens, 0);
        assert!(result.pages.is_empty());
    }

    #[tokio::test]
    async fn test_insert_and_match() {
        let cache = PromptCache::new(1000);
        let tokens = vec![1i32, 2, 3, 4, 5];
        let pages = vec![10usize, 11, 12, 13, 14];
        cache.insert(&tokens, &pages).await.unwrap();

        // Exact prefix match
        let result = cache.match_prefix(&[1, 2, 3, 4, 5, 6]).await;
        assert_eq!(result.matched_tokens, 5);
        assert_eq!(result.pages, pages);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let cache = PromptCache::new(5);
        cache.insert(&[1, 2, 3], &[1, 2, 3]).await.unwrap();
        // Inserting 4 more pages should trigger eviction (capacity = 5)
        cache.insert(&[4, 5, 6, 7], &[4, 5, 6, 7]).await.unwrap();
        // Cache should not exceed max_pages after eviction
        assert!(cache.cached_pages().await <= 5);
    }
}
