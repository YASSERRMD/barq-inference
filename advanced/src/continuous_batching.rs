//! Continuous batching for multi-request inference
//!
//! Allows multiple requests to share GPU compute per forward pass.
//! Near-linear throughput scaling for multi-user workloads.
//!
//! Expected gain: 4-6x aggregate TPS at 8 concurrent requests

use std::collections::VecDeque;

use barq_core::error::{Error, Result};

/// Active sequence processing state
#[derive(Debug)]
pub struct ActiveSequence {
    pub seq_id: i32,
    pub prompt_tokens: Vec<i32>,
    pub generated_tokens: Vec<i32>,
    pub max_tokens: usize,
    pub next_pos: i32,
    pub is_prefill: bool,
    pub response_tx: tokio::sync::oneshot::Sender<Vec<i32>>,
}

/// Batch scheduler for continuous batching
pub struct BatchScheduler {
    /// Pending requests waiting to be admitted
    pending: VecDeque<PendingRequest>,
    /// Active sequences being processed
    active: Vec<ActiveSequence>,
    /// Maximum batch size
    max_batch: usize,
    /// Maximum concurrent sequences
    max_sequences: usize,
    /// Total context budget (sum of all sequence positions)
    n_ctx_total: usize,
}

/// Pending request
pub struct PendingRequest {
    pub tokens: Vec<i32>,
    pub max_tokens: usize,
    pub response_tx: tokio::sync::oneshot::Sender<Vec<i32>>,
}

impl BatchScheduler {
    /// Create new batch scheduler
    pub fn new(max_batch: usize, max_sequences: usize, n_ctx_total: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            active: Vec::new(),
            max_batch,
            max_sequences,
            n_ctx_total,
        }
    }

    /// Add a new request to the pending queue
    pub async fn add_request(&mut self, tokens: Vec<i32>, max_tokens: usize) -> Result<Vec<i32>> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        self.pending.push_back(PendingRequest {
            tokens,
            max_tokens,
            response_tx,
        });

        // Return response channel
        tokio::task::spawn(async move {
            match response_rx.await {
                Ok(tokens) => Ok(tokens),
                Err(_) => Err(Error::Backend("Request cancelled".to_string())),
            }
        })
        .await
        .unwrap()
    }

    /// Fill batch for next forward pass
    pub fn fill_batch(&mut self) -> Batch {
        let mut batch_tokens = Vec::new();
        let mut batch_positions = Vec::new();
        let mut batch_seq_id = Vec::new();

        // Admit from pending if space is available and batch is not full
        while self.active.len() < self.max_sequences && !self.pending.is_empty() {
            if let Some(pending) = self.pending.pop_front() {
                // Find unused seq_id
                let used_ids: std::collections::HashSet<i32> =
                    self.active.iter().map(|s| s.seq_id).collect();
                let mut seq_id = 0;
                while used_ids.contains(&seq_id) {
                    seq_id += 1;
                }

                self.active.push(ActiveSequence {
                    seq_id,
                    prompt_tokens: pending.tokens,
                    generated_tokens: Vec::new(),
                    max_tokens: pending.max_tokens,
                    next_pos: 0,
                    is_prefill: true,
                    response_tx: pending.response_tx,
                });
            }
        }

        // Build actual decode/prefill lists
        for seq in &mut self.active {
            if seq.is_prefill {
                // Prefill: push entire prompt
                for &token in &seq.prompt_tokens {
                    batch_tokens.push(token);
                    batch_positions.push(seq.next_pos);
                    batch_seq_id.push(seq.seq_id);
                    seq.next_pos += 1;
                }
                seq.is_prefill = false;
            } else {
                // Decode: push only the last generated token
                if let Some(&last_token) = seq.generated_tokens.last() {
                    batch_tokens.push(last_token);
                    batch_positions.push(seq.next_pos);
                    batch_seq_id.push(seq.seq_id);
                    seq.next_pos += 1;
                }
            }
        }

        Batch {
            tokens: batch_tokens,
            positions: batch_positions,
            seq_ids: batch_seq_id,
        }
    }

    /// Update sequences after forward pass
    pub fn update_sequences(&mut self, new_tokens: &[i32]) -> Result<()> {
        if new_tokens.len() != self.active.len() {
            return Err(Error::Backend(format!(
                "Batch mismatch: {} new tokens for {} active sequences",
                new_tokens.len(),
                self.active.len(),
            )));
        }

        let mut completed_indices = Vec::new();

        for (idx, seq) in self.active.iter_mut().enumerate() {
            let token = new_tokens[idx];
            seq.generated_tokens.push(token);

            // Check if sequence is complete (EOS or max_tokens reached)
            if token == 0 || token == 2 || seq.generated_tokens.len() >= seq.max_tokens {
                completed_indices.push(idx);
            }
        }

        // Remove completed sequences in reverse order to preserve indices
        for &idx in completed_indices.iter().rev() {
            let seq = self.active.remove(idx);
            let _ = seq.response_tx.send(seq.generated_tokens);
        }

        Ok(())
    }

    /// Get number of active sequences
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Get pending queue length
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Batch for forward pass
#[derive(Debug, Clone)]
pub struct Batch {
    pub tokens: Vec<i32>,
    pub positions: Vec<i32>,
    pub seq_ids: Vec<i32>,
}

impl Batch {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            positions: Vec::new(),
            seq_ids: Vec::new(),
        }
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new()
    }
}

/// Continuous batching configuration
#[derive(Debug, Clone)]
pub struct ContinuousBatchingConfig {
    /// Maximum batch size (recommended: 8-32)
    pub max_batch: usize,
    /// Maximum concurrent sequences
    pub max_sequences: usize,
    /// Total context budget
    pub n_ctx_total: usize,
}

impl Default for ContinuousBatchingConfig {
    fn default() -> Self {
        Self {
            max_batch: 2048,    // Max tokens per decode step
            max_sequences: 32,  // Max concurrent sequences
            n_ctx_total: 32768, // Total context for all sequences
        }
    }
}

impl ContinuousBatchingConfig {
    /// Optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_batch: 512,
            max_sequences: 8,
            n_ctx_total: 16384,
        }
    }

    /// Optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_batch: 2048,
            max_sequences: 32,
            n_ctx_total: 65536,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_scheduler_creation() {
        let scheduler = BatchScheduler::new(2048, 32, 32768);
        assert_eq!(scheduler.max_batch, 2048);
        assert_eq!(scheduler.max_sequences, 32);
        assert_eq!(scheduler.n_ctx_total, 32768);
    }

    #[test]
    fn test_batch_creation() {
        let batch = Batch::new();
        assert_eq!(batch.token_count(), 0);
    }

    #[test]
    fn test_config_defaults() {
        let config = ContinuousBatchingConfig::default();
        assert_eq!(config.max_batch, 2048);
        assert_eq!(config.max_sequences, 32);
    }

    #[test]
    fn test_config_low_latency() {
        let config = ContinuousBatchingConfig::low_latency();
        assert_eq!(config.max_batch, 512);
        assert_eq!(config.max_sequences, 8);
    }

    #[test]
    fn test_config_high_throughput() {
        let config = ContinuousBatchingConfig::high_throughput();
        assert_eq!(config.max_batch, 2048);
        assert_eq!(config.max_sequences, 32);
        assert_eq!(config.n_ctx_total, 65536);
    }
}
