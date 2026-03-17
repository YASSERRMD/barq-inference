//! Async inference server with Unix Domain Socket transport
//!
//! Replaces HTTP with UDS for lower latency in local deployments.
//! Eliminates TCP overhead for agentic loops and local inference.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, oneshot, Semaphore};

use barq_core::error::{Error, Result};

/// Inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
}

/// Inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub id: String,
    pub text: String,
    pub tokens_generated: usize,
    pub ttft_ms: u64,
    pub total_time_ms: u64,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Socket path for Unix Domain Socket
    pub socket_path: PathBuf,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Request queue size
    pub queue_size: usize,
    /// Number of worker threads
    pub worker_threads: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/tmp/barq-inference.sock"),
            max_concurrent: 8,
            queue_size: 100,
            worker_threads: 4,
        }
    }
}

/// Async inference server
pub struct InferenceServer {
    config: ServerConfig,
    request_tx: mpsc::Sender<InferenceTask>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

/// Internal task for request processing
struct InferenceTask {
    request: InferenceRequest,
    response_tx: oneshot::Sender<Result<InferenceResponse>>,
}

impl InferenceServer {
    /// Create new inference server
    pub fn new(config: ServerConfig) -> Self {
        let (request_tx, _) = mpsc::channel(config.queue_size);

        Self {
            config,
            request_tx,
            shutdown_tx: None,
        }
    }

    /// Start the server
    pub async fn start(&mut self) -> Result<()> {
        // Remove existing socket if present
        let _ = std::fs::remove_file(&self.config.socket_path);

        let listener = UnixListener::bind(&self.config.socket_path)
            .map_err(|e| Error::Backend(format!("Failed to bind UDS: {}", e)))?;

        println!(
            "Inference server listening on: {:?}",
            self.config.socket_path
        );

        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);

        // Spawn request processor
        // TODO: Fix channel handling
        // let request_rx = self.request_tx.clone();
        // tokio::spawn(async move {
        //     Self::request_processor(request_rx).await;
        // });

        // Accept connections
        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, _)) => {
                            let request_tx = self.request_tx.clone();
                            tokio::spawn(async move {
                                Self::handle_connection(stream, request_tx).await;
                            });
                        }
                        Err(e) => {
                            eprintln!("Connection error: {}", e);
                        }
                    }
                }
                _ = &mut shutdown_rx => {
                    println!("Server shutdown signal received");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle client connection
    async fn handle_connection(mut stream: UnixStream, request_tx: mpsc::Sender<InferenceTask>) {
        let mut buffer = vec![0u8; 8192];

        loop {
            // Read request length
            let mut len_buf = [0u8; 4];
            if let Err(e) = stream.read_exact(&mut len_buf).await {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                eprintln!("Read error: {}", e);
                break;
            }

            let len = u32::from_be_bytes(len_buf) as usize;
            if len > buffer.len() {
                buffer.resize(len, 0);
            }

            // Read request data
            if let Err(e) = stream.read_exact(&mut buffer[..len]).await {
                eprintln!("Read error: {}", e);
                break;
            }

            // Parse request
            let request: InferenceRequest = match serde_json::from_slice(&buffer[..len]) {
                Ok(req) => req,
                Err(e) => {
                    eprintln!("Deserialize error: {}", e);
                    break;
                }
            };

            // Create response channel
            let (response_tx, response_rx) = oneshot::channel();

            // Send to request processor
            let task = InferenceTask {
                request: request.clone(),
                response_tx,
            };

            if request_tx.send(task).await.is_err() {
                eprintln!("Request queue full");
                break;
            }

            // Wait for response
            let response = match response_rx.await {
                Ok(Ok(resp)) => resp,
                Ok(Err(e)) => {
                    let error_resp = InferenceResponse {
                        id: request.id,
                        text: format!("Error: {}", e),
                        tokens_generated: 0,
                        ttft_ms: 0,
                        total_time_ms: 0,
                    };
                    error_resp
                }
                Err(_) => {
                    let error_resp = InferenceResponse {
                        id: request.id,
                        text: "Request cancelled".to_string(),
                        tokens_generated: 0,
                        ttft_ms: 0,
                        total_time_ms: 0,
                    };
                    error_resp
                }
            };

            // Send response
            let resp_bytes = match serde_json::to_vec(&response) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("Serialize error: {}", e);
                    break;
                }
            };

            let resp_len = (resp_bytes.len() as u32).to_be_bytes();
            if let Err(e) = stream.write_all(&resp_len).await {
                eprintln!("Write error: {}", e);
                break;
            }

            if let Err(e) = stream.write_all(&resp_bytes).await {
                eprintln!("Write error: {}", e);
                break;
            }
        }
    }

    /// Request processor (runs on blocking thread pool)
    async fn request_processor(mut rx: mpsc::Receiver<InferenceTask>) {
        while let Some(task) = rx.recv().await {
            let InferenceTask {
                request,
                response_tx,
            } = task;

            tokio::task::spawn_blocking(move || {
                let start = std::time::Instant::now();

                // TODO: Implement actual inference
                // For now, simulate
                let response = InferenceResponse {
                    id: request.id.clone(),
                    text: format!("Response to: {}", request.prompt),
                    tokens_generated: 10,
                    ttft_ms: 100,
                    total_time_ms: start.elapsed().as_millis() as u64,
                };

                let _ = response_tx.send(Ok(response));
            });
        }
    }

    /// Shutdown the server
    pub fn shutdown(&mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        Ok(())
    }

    /// Submit inference request
    pub async fn submit(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let (response_tx, response_rx) = oneshot::channel();

        let task = InferenceTask {
            request: request.clone(),
            response_tx,
        };

        self.request_tx
            .send(task)
            .await
            .map_err(|_| Error::Backend("Request queue full".to_string()))?;

        response_rx
            .await
            .map_err(|_| Error::Backend("Request cancelled".to_string()))?
    }
}

/// Client for connecting to inference server
pub struct InferenceClient {
    socket_path: PathBuf,
}

impl InferenceClient {
    /// Create new client
    pub fn new(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    /// Connect to server and send inference request
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let mut stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| Error::Backend(format!("Connection failed: {}", e)))?;

        // Serialize request
        let req_bytes = serde_json::to_vec(&request)
            .map_err(|e| Error::Backend(format!("Serialize error: {}", e)))?;

        // Send length
        let len = (req_bytes.len() as u32).to_be_bytes();
        stream
            .write_all(&len)
            .await
            .map_err(|e| Error::Backend(format!("Write error: {}", e)))?;

        // Send data
        stream
            .write_all(&req_bytes)
            .await
            .map_err(|e| Error::Backend(format!("Write error: {}", e)))?;

        // Read response length
        let mut len_buf = [0u8; 4];
        stream
            .read_exact(&mut len_buf)
            .await
            .map_err(|e| Error::Backend(format!("Read error: {}", e)))?;

        let resp_len = u32::from_be_bytes(len_buf) as usize;

        // Read response data
        let mut buffer = vec![0u8; resp_len];
        stream
            .read_exact(&mut buffer)
            .await
            .map_err(|e| Error::Backend(format!("Read error: {}", e)))?;

        // Deserialize response
        let response: InferenceResponse = serde_json::from_slice(&buffer)
            .map_err(|e| Error::Backend(format!("Deserialize error: {}", e)))?;

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_concurrent, 8);
        assert_eq!(config.queue_size, 100);
    }

    #[tokio::test]
    async fn test_client_creation() {
        let client = InferenceClient::new(PathBuf::from("/tmp/test.sock"));
        assert_eq!(client.socket_path, PathBuf::from("/tmp/test.sock"));
    }
}
