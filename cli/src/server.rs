//! OpenAI-compatible HTTP server for Barq.
//!
//! Phase 24.2 adds the basic serving surface:
//! - `/v1/completions`
//! - `/v1/chat/completions`
//! - Server-Sent Events streaming
//! - CORS support

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use bytes::Bytes;
use futures_util::stream;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{
    ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
    CACHE_CONTROL, CONNECTION, CONTENT_TYPE,
};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use models::{
    context::{Batch, ContextParams},
    llama::LlamaModel,
    loader::Model,
    LlmArch,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::info;
use vocab::{ChatMessage, ChatRole, ChatTemplate, GgufTokenizer, Tokenizer};

type RespBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// Configuration for the HTTP server.
#[derive(Debug, Clone)]
pub struct HttpServerConfig {
    pub model_path: PathBuf,
    pub host: String,
    pub port: u16,
    pub context_size: usize,
    pub rate_limit_rpm: u32,
}

#[derive(Debug, Clone, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
struct ChatMessageResponse {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageResponse,
    finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
struct ChatChunkChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Clone, Deserialize)]
struct IncomingMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionsRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<IncomingMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionsRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Clone)]
struct GenerationOptions {
    max_tokens: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
}

#[derive(Debug, Clone)]
struct CompletionResult {
    prompt_tokens: usize,
    generated_ids: Vec<u32>,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
struct ModelCard {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Debug, Clone, Serialize)]
struct ModelListResponse {
    object: String,
    data: Vec<ModelCard>,
}

#[derive(Debug, Clone, Serialize)]
struct ResponsesResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    output: String,
    usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    #[serde(alias = "prompt")]
    input: Option<String>,
    #[serde(default)]
    messages: Option<Vec<IncomingMessage>>,
    #[serde(default, alias = "max_output_tokens", alias = "max_tokens")]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug)]
struct RateLimitState {
    tokens: f64,
    last_refill: Instant,
}

#[derive(Debug)]
struct RateLimiter {
    capacity: f64,
    refill_per_second: f64,
    states: Mutex<HashMap<String, RateLimitState>>,
}

impl RateLimiter {
    fn new(requests_per_minute: u32) -> Self {
        let capacity = requests_per_minute.max(1) as f64;
        Self {
            capacity,
            refill_per_second: capacity / 60.0,
            states: Mutex::new(HashMap::new()),
        }
    }

    async fn allow(&self, key: &str) -> bool {
        let mut states = self.states.lock().await;
        let state = states
            .entry(key.to_string())
            .or_insert_with(|| RateLimitState {
                tokens: self.capacity,
                last_refill: Instant::now(),
            });

        let elapsed = state.last_refill.elapsed().as_secs_f64();
        let refill = elapsed * self.refill_per_second;
        if refill > 0.0 {
            state.tokens = (state.tokens + refill).min(self.capacity);
            state.last_refill = Instant::now();
        }

        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

struct HttpServerState {
    model_path: PathBuf,
    model_id: String,
    model_arch: LlmArch,
    llama_model: Arc<LlamaModel>,
    tokenizer: Arc<GgufTokenizer>,
    chat_template: ChatTemplate,
    context_size: usize,
    rate_limiter: Option<RateLimiter>,
}

/// Run the HTTP server.
pub async fn run_http_server(config: HttpServerConfig) -> anyhow::Result<()> {
    let loaded_model = Model::load(&config.model_path).await?;
    let model_id = config
        .model_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("barq-model")
        .to_string();
    let model_arch = loaded_model.arch();
    let model_arc = Arc::new(loaded_model);
    let llama_model = Arc::new(LlamaModel::new(model_arc.clone())?);
    let tokenizer = Arc::new(GgufTokenizer::from_gguf(model_arc.metadata()));
    let chat_template = ChatTemplate::for_arch(model_arch.name());

    let state = Arc::new(HttpServerState {
        model_path: config.model_path.clone(),
        model_id: model_id.clone(),
        model_arch,
        llama_model,
        tokenizer,
        chat_template,
        context_size: config.context_size,
        rate_limiter: Some(RateLimiter::new(config.rate_limit_rpm)),
    });

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let listener = TcpListener::bind(addr).await?;

    info!(
        "OpenAI-compatible HTTP server listening on http://{} (model: {}, arch: {:?})",
        addr, model_id, model_arch
    );
    info!(
        "Phase 24.3 server is ready: /v1/models, /v1/responses, /v1/completions and /v1/chat/completions"
    );
    info!(
        "Rate limiting enabled at {} requests/minute per client",
        config.rate_limit_rpm
    );

    loop {
        let (stream, peer) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let state = state.clone();

        tokio::spawn(async move {
            let service = service_fn(move |req| {
                let state = state.clone();
                async move { Ok::<_, Infallible>(state.handle(req, peer).await) }
            });

            let http = http1::Builder::new();
            let _ = http.serve_connection(io, service).await;
        });
    }
}

impl HttpServerState {
    async fn handle(&self, req: Request<Incoming>, peer: SocketAddr) -> Response<RespBody> {
        match (req.method(), req.uri().path()) {
            (&Method::OPTIONS, _) => self.empty_response(StatusCode::NO_CONTENT),
            (&Method::GET, "/") => self.root_response(),
            (&Method::GET, "/v1/models") => self.models_response(),
            (&Method::POST, "/v1/completions") => self.handle_completions(req, peer).await,
            (&Method::POST, "/v1/chat/completions") => {
                self.handle_chat_completions(req, peer).await
            }
            (&Method::POST, "/v1/responses") => self.handle_responses(req, peer).await,
            _ => self.error_response(
                StatusCode::NOT_FOUND,
                "route not found; try /v1/models, /v1/responses, /v1/completions or /v1/chat/completions",
            ),
        }
    }

    fn root_response(&self) -> Response<RespBody> {
        let body = json!({
            "object": "server.info",
            "model": self.model_id,
            "architecture": self.model_arch.name(),
            "endpoints": [
                "/v1/models",
                "/v1/responses",
                "/v1/completions",
                "/v1/chat/completions",
            ],
        });
        self.json_response(StatusCode::OK, body)
    }

    fn models_response(&self) -> Response<RespBody> {
        self.json_response(
            StatusCode::OK,
            ModelListResponse {
                object: "list".to_string(),
                data: vec![ModelCard {
                    id: self.model_id.clone(),
                    object: "model".to_string(),
                    created: unix_timestamp(),
                    owned_by: "barq".to_string(),
                }],
            },
        )
    }

    async fn handle_completions(
        &self,
        req: Request<Incoming>,
        peer: SocketAddr,
    ) -> Response<RespBody> {
        if let Some(resp) = self.enforce_rate_limit(peer).await {
            return resp;
        }

        let request: CompletionsRequest = match self.read_json(req).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };

        let options = GenerationOptions {
            max_tokens: request.max_tokens.unwrap_or(128),
            temperature: request.temperature.unwrap_or(0.8),
            top_k: request.top_k.unwrap_or(40),
            top_p: request.top_p.unwrap_or(0.95),
        };

        match self.generate_text(request.prompt, options).await {
            Ok(result) => {
                if request.stream.unwrap_or(false) {
                    self.stream_completion_response(result, request.model, peer)
                        .await
                } else {
                    self.completion_response(result, request.model)
                }
            }
            Err(err) => self.error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
        }
    }

    async fn handle_chat_completions(
        &self,
        req: Request<Incoming>,
        peer: SocketAddr,
    ) -> Response<RespBody> {
        if let Some(resp) = self.enforce_rate_limit(peer).await {
            return resp;
        }

        let request: ChatCompletionsRequest = match self.read_json(req).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };

        let (system_prompt, messages) = match self.normalize_messages(&request.messages) {
            Ok(value) => value,
            Err(resp) => return resp,
        };

        let prompt = match self.render_chat_prompt(system_prompt.as_deref(), &messages) {
            Ok(prompt) => prompt,
            Err(err) => return self.error_response(StatusCode::BAD_REQUEST, &err.to_string()),
        };

        let options = GenerationOptions {
            max_tokens: request.max_tokens.unwrap_or(128),
            temperature: request.temperature.unwrap_or(0.8),
            top_k: request.top_k.unwrap_or(40),
            top_p: request.top_p.unwrap_or(0.95),
        };

        match self.generate_text(prompt, options).await {
            Ok(result) => {
                if request.stream.unwrap_or(false) {
                    self.stream_chat_response(result, request.model, peer).await
                } else {
                    self.chat_response(result, request.model)
                }
            }
            Err(err) => self.error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
        }
    }

    async fn handle_responses(
        &self,
        req: Request<Incoming>,
        peer: SocketAddr,
    ) -> Response<RespBody> {
        if let Some(resp) = self.enforce_rate_limit(peer).await {
            return resp;
        }

        let request: ResponsesRequest = match self.read_json(req).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };

        let prompt = match self.render_responses_prompt(&request) {
            Ok(prompt) => prompt,
            Err(resp) => return resp,
        };

        let options = GenerationOptions {
            max_tokens: request.max_tokens.unwrap_or(128),
            temperature: request.temperature.unwrap_or(0.8),
            top_k: request.top_k.unwrap_or(40),
            top_p: request.top_p.unwrap_or(0.95),
        };

        match self.generate_text(prompt, options).await {
            Ok(result) => {
                if request.stream.unwrap_or(false) {
                    self.stream_responses_response(result, request.model, peer)
                        .await
                } else {
                    self.responses_response(result, request.model)
                }
            }
            Err(err) => self.error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
        }
    }

    fn normalize_messages(
        &self,
        messages: &[IncomingMessage],
    ) -> Result<(Option<String>, Vec<ChatMessage>), Response<RespBody>> {
        let mut system_prompt = Vec::new();
        let mut chat_messages = Vec::new();

        for message in messages {
            let role = parse_role(&message.role).ok_or_else(|| {
                self.error_response(
                    StatusCode::BAD_REQUEST,
                    &format!("unsupported message role `{}`", message.role),
                )
            })?;

            if role == ChatRole::System {
                system_prompt.push(message.content.clone());
            } else {
                chat_messages.push(ChatMessage::new(role, message.content.clone()));
            }
        }

        let system_prompt = if system_prompt.is_empty() {
            None
        } else {
            Some(system_prompt.join("\n"))
        };

        Ok((system_prompt, chat_messages))
    }

    fn render_responses_prompt(
        &self,
        request: &ResponsesRequest,
    ) -> Result<String, Response<RespBody>> {
        if let Some(messages) = &request.messages {
            let (system_prompt, chat_messages) = self.normalize_messages(messages)?;
            return self
                .render_chat_prompt(system_prompt.as_deref(), &chat_messages)
                .map_err(|err| self.error_response(StatusCode::BAD_REQUEST, &err.to_string()));
        }

        if let Some(input) = &request.input {
            return Ok(input.clone());
        }

        Err(self.error_response(
            StatusCode::BAD_REQUEST,
            "responses request requires `input` or `messages`",
        ))
    }

    fn render_chat_prompt(
        &self,
        system_prompt: Option<&str>,
        messages: &[ChatMessage],
    ) -> anyhow::Result<String> {
        self.chat_template
            .render(Some(self.tokenizer.vocab()), system_prompt, messages)
            .context("failed to render chat prompt")
    }

    async fn generate_text(
        &self,
        prompt: String,
        options: GenerationOptions,
    ) -> anyhow::Result<CompletionResult> {
        let params = ContextParams {
            n_ctx: self.context_size as u32,
            n_threads: num_cpus::get_physical() as u32,
            n_threads_batch: num_cpus::get_physical() as u32,
            ..Default::default()
        };

        let (prompt_tokens, prompt_token_count) =
            tokenize_prompt(self.tokenizer.as_ref(), &prompt, true).await?;

        if prompt_tokens.is_empty() {
            return Err(anyhow::anyhow!("prompt tokenized to no tokens"));
        }

        let context = self.llama_model.create_context(params)?;
        let generated_ids = context
            .generate(
                &prompt_tokens,
                options.max_tokens,
                options.temperature,
                options.top_k,
                options.top_p,
            )
            .await?;

        let eos = self.tokenizer.vocab().special_tokens().eos;
        let generated_ids: Vec<u32> = generated_ids
            .into_iter()
            .filter(|&id| Some(id as u32) != eos)
            .map(|id| id as u32)
            .collect();
        let text = self.tokenizer.decode(&generated_ids).await?;

        Ok(CompletionResult {
            prompt_tokens: prompt_token_count,
            generated_ids,
            text,
        })
    }

    fn completion_response(
        &self,
        result: CompletionResult,
        requested_model: Option<String>,
    ) -> Response<RespBody> {
        let model = requested_model.unwrap_or_else(|| self.model_id.clone());
        let response = CompletionResponse {
            id: response_id("cmpl"),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model,
            choices: vec![CompletionChoice {
                index: 0,
                text: result.text.clone(),
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.generated_ids.len(),
                total_tokens: result.prompt_tokens + result.generated_ids.len(),
            },
        };
        self.json_response(StatusCode::OK, response)
    }

    fn responses_response(
        &self,
        result: CompletionResult,
        requested_model: Option<String>,
    ) -> Response<RespBody> {
        let model = requested_model.unwrap_or_else(|| self.model_id.clone());
        let response = ResponsesResponse {
            id: response_id("resp"),
            object: "response".to_string(),
            created: unix_timestamp(),
            model,
            output: result.text.clone(),
            usage: Usage {
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.generated_ids.len(),
                total_tokens: result.prompt_tokens + result.generated_ids.len(),
            },
        };
        self.json_response(StatusCode::OK, response)
    }

    fn chat_response(
        &self,
        result: CompletionResult,
        requested_model: Option<String>,
    ) -> Response<RespBody> {
        let model = requested_model.unwrap_or_else(|| self.model_id.clone());
        let response = ChatCompletionResponse {
            id: response_id("chatcmpl"),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content: result.text.clone(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.generated_ids.len(),
                total_tokens: result.prompt_tokens + result.generated_ids.len(),
            },
        };
        self.json_response(StatusCode::OK, response)
    }

    async fn stream_completion_response(
        &self,
        result: CompletionResult,
        requested_model: Option<String>,
        _peer: SocketAddr,
    ) -> Response<RespBody> {
        let model = requested_model.unwrap_or_else(|| self.model_id.clone());
        let created = unix_timestamp();
        let id = response_id("cmpl");
        let model_name = model.clone();
        let event_id = id.clone();

        let decoded_tokens = self
            .decode_token_chunks(&result.generated_ids)
            .await
            .unwrap_or_default();
        let chunks = decoded_tokens
            .into_iter()
            .map(move |content| {
                let chunk = json!({
                    "id": event_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "text": content,
                        "finish_reason": null,
                    }],
                });
                Ok::<_, Infallible>(Frame::data(Bytes::from(format!("data: {}\n\n", chunk))))
            })
            .chain(std::iter::once(Ok(Frame::data(Bytes::from(
                "data: [DONE]\n\n",
            )))));

        self.stream_response(chunks)
    }

    async fn stream_chat_response(
        &self,
        result: CompletionResult,
        requested_model: Option<String>,
        _peer: SocketAddr,
    ) -> Response<RespBody> {
        let model = requested_model.unwrap_or_else(|| self.model_id.clone());
        let created = unix_timestamp();
        let id = response_id("chatcmpl");
        let model_name = model.clone();
        let event_id = id.clone();

        let decoded_tokens = self
            .decode_token_chunks(&result.generated_ids)
            .await
            .unwrap_or_default();

        let chunks = std::iter::once(Ok::<_, Infallible>(Frame::data(Bytes::from(format!(
            "data: {}\n\n",
            json!({
                "id": event_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": null,
                }],
            })
        )))))
        .chain(decoded_tokens.into_iter().map(move |content| {
            Ok::<_, Infallible>(Frame::data(Bytes::from(format!(
                "data: {}\n\n",
                json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": null,
                    }],
                })
            ))))
        }))
        .chain(std::iter::once(Ok(Frame::data(Bytes::from(
            "data: [DONE]\n\n",
        )))));

        self.stream_response(chunks)
    }

    async fn stream_responses_response(
        &self,
        result: CompletionResult,
        requested_model: Option<String>,
        _peer: SocketAddr,
    ) -> Response<RespBody> {
        let model = requested_model.unwrap_or_else(|| self.model_id.clone());
        let created = unix_timestamp();
        let id = response_id("resp");
        let model_name = model.clone();
        let event_id = id.clone();

        let decoded_tokens = self
            .decode_token_chunks(&result.generated_ids)
            .await
            .unwrap_or_default();

        let chunks = decoded_tokens
            .into_iter()
            .map(move |content| {
                let chunk = json!({
                    "id": event_id,
                    "object": "response.chunk",
                    "created": created,
                    "model": model_name,
                    "output": content,
                });
                Ok::<_, Infallible>(Frame::data(Bytes::from(format!("data: {}\n\n", chunk))))
            })
            .chain(std::iter::once(Ok(Frame::data(Bytes::from(
                "data: [DONE]\n\n",
            )))));

        self.stream_response(chunks)
    }

    async fn decode_token_chunks(&self, tokens: &[u32]) -> anyhow::Result<Vec<String>> {
        let mut decoded = Vec::with_capacity(tokens.len());
        for token in tokens {
            decoded.push(self.tokenizer.decode(&[*token]).await?);
        }
        Ok(decoded)
    }

    async fn enforce_rate_limit(&self, peer: SocketAddr) -> Option<Response<RespBody>> {
        let limiter = match &self.rate_limiter {
            Some(limiter) => limiter,
            None => return None,
        };

        if limiter.allow(&peer.ip().to_string()).await {
            None
        } else {
            Some(self.rate_limit_response())
        }
    }

    fn stream_response<I>(&self, chunks: I) -> Response<RespBody>
    where
        I: Iterator<Item = Result<Frame<Bytes>, Infallible>> + Send + Sync + 'static,
    {
        let stream = stream::iter(chunks);
        let body = StreamBody::new(stream).boxed();
        let mut response = Response::new(body);
        let headers = response.headers_mut();
        headers.insert(
            CONTENT_TYPE,
            "text/event-stream; charset=utf-8".parse().unwrap(),
        );
        headers.insert(CACHE_CONTROL, "no-cache".parse().unwrap());
        headers.insert(CONNECTION, "keep-alive".parse().unwrap());
        self.add_cors_headers(response)
    }

    fn empty_response(&self, status: StatusCode) -> Response<RespBody> {
        let mut response = Response::new(Full::new(Bytes::new()).boxed());
        *response.status_mut() = status;
        self.add_cors_headers(response)
    }

    fn json_response<T: Serialize>(&self, status: StatusCode, value: T) -> Response<RespBody> {
        let body = match serde_json::to_vec(&value) {
            Ok(bytes) => Full::new(Bytes::from(bytes)).boxed(),
            Err(err) => {
                return self.error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("serialization error: {}", err),
                );
            }
        };

        let mut response = Response::new(body);
        *response.status_mut() = status;
        response
            .headers_mut()
            .insert(CONTENT_TYPE, "application/json".parse().unwrap());
        self.add_cors_headers(response)
    }

    fn error_response(&self, status: StatusCode, message: &str) -> Response<RespBody> {
        let response = ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                kind: "invalid_request_error".to_string(),
            },
        };
        let mut resp = self.json_response(status, response);
        *resp.status_mut() = status;
        resp
    }

    fn rate_limit_response(&self) -> Response<RespBody> {
        let response = ErrorResponse {
            error: ErrorDetail {
                message: "rate limit exceeded".to_string(),
                kind: "rate_limit_error".to_string(),
            },
        };
        let mut resp = self.json_response(StatusCode::TOO_MANY_REQUESTS, response);
        *resp.status_mut() = StatusCode::TOO_MANY_REQUESTS;
        resp
    }

    fn add_cors_headers(&self, mut response: Response<RespBody>) -> Response<RespBody> {
        let headers = response.headers_mut();
        headers.insert(ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
        headers.insert(
            ACCESS_CONTROL_ALLOW_HEADERS,
            "content-type".parse().unwrap(),
        );
        headers.insert(
            ACCESS_CONTROL_ALLOW_METHODS,
            "GET, POST, OPTIONS".parse().unwrap(),
        );
        response
    }

    async fn read_json<T: for<'de> Deserialize<'de>>(
        &self,
        req: Request<Incoming>,
    ) -> Result<T, Response<RespBody>> {
        let bytes = req
            .into_body()
            .collect()
            .await
            .map_err(|err| self.error_response(StatusCode::BAD_REQUEST, &err.to_string()))?
            .to_bytes();

        serde_json::from_slice(&bytes)
            .map_err(|err| self.error_response(StatusCode::BAD_REQUEST, &err.to_string()))
    }
}

async fn tokenize_prompt<T: Tokenizer>(
    tokenizer: &T,
    text: &str,
    add_special: bool,
) -> anyhow::Result<(Vec<i32>, usize)> {
    let tokenization_result = tokenizer.tokenize(text, add_special).await?;
    let ids: Vec<i32> = tokenization_result
        .ids
        .iter()
        .map(|&id| id as i32)
        .collect();
    let count = tokenization_result.ids.len();
    Ok((ids, count))
}

fn parse_role(role: &str) -> Option<ChatRole> {
    match role.to_ascii_lowercase().as_str() {
        "system" => Some(ChatRole::System),
        "user" => Some(ChatRole::User),
        "assistant" => Some(ChatRole::Assistant),
        "tool" => Some(ChatRole::Tool),
        "developer" => Some(ChatRole::Developer),
        _ => None,
    }
}

fn response_id(prefix: &str) -> String {
    format!("{}-{}", prefix, unix_timestamp())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vocab::tokenizer::WhitespaceTokenizer;

    #[tokio::test]
    async fn tokenize_prompt_counts_tokens() {
        let tokenizer = WhitespaceTokenizer::new();
        let (_ids, count) = tokenize_prompt(&tokenizer, "hello world", false)
            .await
            .expect("tokenization should succeed");

        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn rate_limiter_allows_then_blocks() {
        let limiter = RateLimiter::new(1);

        assert!(limiter.allow("127.0.0.1").await);
        assert!(!limiter.allow("127.0.0.1").await);
    }

    #[test]
    fn responses_request_accepts_prompt_alias() {
        let request: ResponsesRequest =
            serde_json::from_str(r#"{"prompt":"Say hello"}"#).expect("request should deserialize");

        assert_eq!(request.input.as_deref(), Some("Say hello"));
    }
}
