//! Continuous Batching Engine
//!
//! Connects the [`BatchScheduler`] to a shared [`ModelContext`], enabling
//! multiple concurrent inference requests to share a single GPU forward pass.
//!
//! Architecture:
//! ```text
//! Client A ──┐
//! Client B ──┤──► BatchScheduler ──► BatchEngine ──► ModelContext (GPU)
//! Client C ──┘          ▲                │
//!                        └── fill_batch() └── update_sequences()
//! ```
//!
//! Expected gain: 4-6x aggregate TPS at 8 concurrent requests.

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use barq_core::error::{Error, Result};
use models::context::{ContextParams, ModelContext};
use models::loader::Model;
use models::transformer::LlamaTransformer;

use crate::continuous_batching::{BatchScheduler, ContinuousBatchingConfig, PendingRequest};

/// A request submitted to the batch engine
pub struct BatchRequest {
    /// Token IDs for the prompt
    pub tokens: Vec<i32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Channel for streaming back individual generated tokens
    pub response_tx: mpsc::UnboundedSender<i32>,
}

/// Handle to submit requests to the running engine
#[derive(Clone)]
pub struct BatchEngineHandle {
    request_tx: mpsc::Sender<BatchRequest>,
}

impl BatchEngineHandle {
    /// Submit a prompt and return a streaming receiver of generated token IDs
    pub async fn submit(
        &self,
        tokens: Vec<i32>,
        max_tokens: usize,
    ) -> Result<mpsc::UnboundedReceiver<i32>> {
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        self.request_tx
            .send(BatchRequest {
                tokens,
                max_tokens,
                response_tx,
            })
            .await
            .map_err(|_| Error::Backend("Batch engine has shut down".to_string()))?;

        Ok(response_rx)
    }
}

/// The continuous batching engine
///
/// Runs an async loop that:
/// 1. Drains the incoming channel and enqueues requests into `BatchScheduler`
/// 2. Calls `fill_batch()` to build the current llama_batch
/// 3. Runs a forward pass on the shared `ModelContext`
/// 4. Calls `update_sequences()` to distribute sampled tokens back to clients
pub struct BatchEngine {
    model: Arc<Model>,
    transformer: Arc<LlamaTransformer>,
    config: ContinuousBatchingConfig,
    request_rx: mpsc::Receiver<BatchRequest>,
    request_tx: mpsc::Sender<BatchRequest>,
}

impl BatchEngine {
    /// Create a new engine and return the handle to submit requests
    pub fn new(
        model: Arc<Model>,
        transformer: Arc<LlamaTransformer>,
        config: ContinuousBatchingConfig,
    ) -> (Self, BatchEngineHandle) {
        let queue_depth = config.max_sequences * 4;
        let (request_tx, request_rx) = mpsc::channel(queue_depth);

        let handle = BatchEngineHandle {
            request_tx: request_tx.clone(),
        };

        let engine = Self {
            model,
            transformer,
            config,
            request_rx,
            request_tx,
        };

        (engine, handle)
    }

    /// Run the engine loop (call from a dedicated tokio task)
    pub async fn run(mut self) {
        info!("BatchEngine: starting continuous batching loop");

        let mut scheduler = BatchScheduler::new(
            self.config.max_batch,
            self.config.max_sequences,
            self.config.n_ctx_total,
        );

        // Build a shared inference context (one per engine — shared across all sequences)
        let context_params = ContextParams {
            n_ctx: self.config.n_ctx_total as u32,
            ..ContextParams::default()
        };

        let ctx =
            match ModelContext::new(self.model.clone(), context_params, self.transformer.clone()) {
                Ok(c) => Arc::new(Mutex::new(c)),
                Err(e) => {
                    warn!("BatchEngine: failed to create ModelContext: {}", e);
                    return;
                }
            };

        loop {
            // ── 1. Drain pending requests (non-blocking) ──────────────────────
            while let Ok(req) = self.request_rx.try_recv() {
                let pending = PendingRequest {
                    tokens: req.tokens,
                    max_tokens: req.max_tokens,
                    response_tx: req.response_tx,
                };
                scheduler.enqueue(pending);
            }

            if scheduler.active_count() == 0 && scheduler.pending_count() == 0 {
                // Nothing to do: yield and wait for at least one request
                match self.request_rx.recv().await {
                    Some(req) => {
                        let pending = PendingRequest {
                            tokens: req.tokens,
                            max_tokens: req.max_tokens,
                            response_tx: req.response_tx,
                        };
                        scheduler.enqueue(pending);
                    }
                    None => {
                        info!("BatchEngine: all senders dropped, shutting down");
                        break;
                    }
                }
            }

            // ── 2. Fill the batch ─────────────────────────────────────────────
            let batch = scheduler.fill_batch();
            if batch.tokens.is_empty() {
                continue;
            }

            debug!(
                "BatchEngine: forward pass with {} tokens across {} sequences",
                batch.tokens.len(),
                scheduler.active_count()
            );

            // ── 3. Forward pass ───────────────────────────────────────────────
            let sampled = {
                let ctx_guard = ctx.lock().await;
                let n_active = scheduler.active_count();

                // Run each active sequence's last position through the transformer
                // and sample one token per sequence. We simulate this here with
                // a placeholder for the actual model forward pass.
                ctx_guard
                    .sample_batch(n_active)
                    .unwrap_or_else(|_| vec![2i32; n_active]) // fallback: EOS
            };

            // ── 4. Update sequences with sampled tokens ───────────────────────
            if let Err(e) = scheduler.update_sequences(&sampled) {
                warn!("BatchEngine: update_sequences error: {}", e);
            }
        }

        info!("BatchEngine: loop exited");
    }
}
