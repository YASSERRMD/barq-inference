//! Paged Attention implementation
//!
//! Paging system for KV cache that allows efficient memory
//! management for very long contexts and batch processing.

use std::collections::HashMap;

use tokio::sync::Mutex;

use barq_core::tensor::{Tensor, TensorType, Shape};
use barq_core::error::{Error, Result};

/// Page in KV cache
#[derive(Debug, Clone)]
pub struct CachePage {
    /// Page ID
    pub id: usize,
    /// Block size (number of tokens per page)
    pub block_size: usize,
    /// Page data (K and V tensors)
    pub data: Option<(Tensor, Tensor)>,
    /// Reference count
    pub ref_count: usize,
}

/// Paged KV cache
pub struct PagedAttention {
    /// Maximum number of pages
    max_pages: usize,
    /// Page block size
    block_size: usize,
    /// Available pages
    free_pages: Mutex<Vec<usize>>,
    /// Allocated pages by sequence
    sequence_pages: Mutex<HashMap<i32, Vec<usize>>>,
    /// Page data
    pages: Mutex<Vec<Option<CachePage>>>,
}

impl PagedAttention {
    /// Create a new paged attention cache
    pub fn new(max_pages: usize, block_size: usize) -> Self {
        let free_pages: Vec<usize> = (0..max_pages).collect();
        let pages: Vec<Option<CachePage>> = (0..max_pages)
            .map(|id| Some(CachePage {
                id,
                block_size,
                data: None,
                ref_count: 0,
            }))
            .collect();

        Self {
            max_pages,
            block_size,
            free_pages: Mutex::new(free_pages),
            sequence_pages: Mutex::new(HashMap::new()),
            pages: Mutex::new(pages),
        }
    }

    /// Allocate pages for a sequence
    pub async fn allocate(&self, seq_id: i32, num_pages: usize) -> Result<Vec<usize>> {
        let mut free_pages = self.free_pages.lock().await;
        let mut sequence_pages = self.sequence_pages.lock().await;

        if free_pages.len() < num_pages {
            return Err(Error::Allocation(format!(
                "Not enough free pages: requested {}, available {}",
                num_pages,
                free_pages.len()
            )));
        }

        let mut allocated = Vec::new();
        for _ in 0..num_pages {
            let page_id = free_pages.pop().unwrap();
            allocated.push(page_id);
        }

        sequence_pages.insert(seq_id, allocated.clone());
        Ok(allocated)
    }

    /// Free pages for a sequence
    pub async fn free(&self, seq_id: i32) -> Result<()> {
        let mut sequence_pages = self.sequence_pages.lock().await;
        let mut free_pages = self.free_pages.lock().await;

        if let Some(pages) = sequence_pages.remove(&seq_id) {
            for page_id in pages {
                free_pages.push(page_id);

                // Clear page data
                let mut all_pages = self.pages.lock().await;
                if let Some(Some(page)) = all_pages.get_mut(page_id) {
                    page.data = None;
                    page.ref_count = 0;
                }
            }
        }

        Ok(())
    }

    /// Get page by ID
    pub async fn get_page(&self, page_id: usize) -> Option<CachePage> {
        let pages = self.pages.lock().await;
        pages.get(page_id).and_then(|p| p.clone())
    }

    /// Write data to a page
    pub async fn write_page(&self, page_id: usize, k: Tensor, v: Tensor) -> Result<()> {
        let mut pages = self.pages.lock().await;

        if let Some(Some(page)) = pages.get_mut(page_id) {
            page.data = Some((k, v));
            page.ref_count += 1;
            Ok(())
        } else {
            Err(Error::Tensor(format!("Invalid page ID: {}", page_id)))
        }
    }

    /// Returns the number of free pages
    pub async fn free_page_count(&self) -> usize {
        self.free_pages.lock().await.len()
    }

    /// Returns the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Defragment memory pages by compacting sparse allocations.
    ///
    /// Scans the page pool and relocates active pages into the lowest available
    /// physical slots, then rebuilds the free list as a contiguous high range.
    /// Updates all `sequence_pages` maps to point to the new physical locations.
    ///
    /// Returns the number of pages actually relocated.
    pub async fn defrag(&self) -> Result<usize> {
        let mut sequence_pages = self.sequence_pages.lock().await;
        let mut free_pages = self.free_pages.lock().await;
        let mut pages = self.pages.lock().await;

        // Build a set of currently used page IDs
        let used: std::collections::HashSet<usize> = sequence_pages
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();

        // Gather free slots in sorted order (lowest first)
        let mut free_slots: Vec<usize> = (0..self.max_pages)
            .filter(|id| !used.contains(id))
            .collect();

        // Gather used slots sorted descending (move high IDs into low slots)
        let mut used_high: Vec<usize> = used.iter()
            .copied()
            .filter(|id| free_slots.first().map(|f| id > f).unwrap_or(false))
            .collect();
        used_high.sort_unstable_by(|a, b| b.cmp(a)); // descending

        // Map old_id -> new_id for relocated pages
        let mut remap: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut pages_moved = 0;

        for old_id in used_high {
            if let Some(&new_id) = free_slots.first() {
                if new_id < old_id {
                    // Extract data from old page before any other mut borrows
                    let old_ref_count = pages[old_id].as_ref().map(|p| p.ref_count).unwrap_or(0);
                    let old_data = pages[old_id].as_mut().map(|p| p.data.take()).flatten();

                    if let Some(data) = old_data {
                        if let Some(Some(new_page)) = pages.get_mut(new_id) {
                            new_page.data = Some(data);
                            new_page.ref_count = old_ref_count;
                        }
                        // Clear old page
                        if let Some(Some(old_page)) = pages.get_mut(old_id) {
                            old_page.data = None;
                            old_page.ref_count = 0;
                        }
                    }
                    remap.insert(old_id, new_id);
                    free_slots.remove(0);
                    free_slots.push(old_id);
                    pages_moved += 1;
                }
            }
        }

        // Update sequence maps
        for page_ids in sequence_pages.values_mut() {
            for id in page_ids.iter_mut() {
                if let Some(&new_id) = remap.get(id) {
                    *id = new_id;
                }
            }
        }

        // Rebuild free list in sorted order
        *free_pages = free_slots;
        free_pages.sort_unstable();

        Ok(pages_moved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_paged_attention() {
        let cache = PagedAttention::new(1024, 16);
        assert_eq!(cache.free_page_count().await, 1024);
        assert_eq!(cache.block_size(), 16);

        let pages = cache.allocate(0, 10).await.unwrap();
        assert_eq!(pages.len(), 10);
        assert_eq!(cache.free_page_count().await, 1014);

        cache.free(0).await.unwrap();
        assert_eq!(cache.free_page_count().await, 1024);
    }
}
