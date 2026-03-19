#![allow(
    clippy::all,
    unexpected_cfgs,
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code
)]
#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code,
    clippy::needless_update,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::let_and_return,
    clippy::manual_range_contains
)]

//! Barq - High-performance LLM inference engine CLI

mod benchmark;
mod performance;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, Level};

#[derive(Parser)]
#[command(name = "barq")]
#[command(about = "Barq - High-performance LLM inference engine", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Number of threads
    #[arg(short, long, global = true, default_value = "4")]
    threads: usize,

    // === Performance Optimizations ===
    /// Enable CUDA Graphs (7-20% TPS gain on NVIDIA GPUs)
    #[arg(long, global = true)]
    cuda_graphs: bool,

    /// Enable Flash Attention (~30% faster, reduces VRAM usage)
    #[arg(long, global = true)]
    flash_attn: bool,

    /// Performance preset: max-speed, balanced, max-quality, cpu, gpu
    #[arg(long, global = true)]
    preset: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model
    Run {
        /// Model path (GGUF file)
        #[arg(short, long)]
        model: PathBuf,

        /// Draft model path for speculative decoding (optional)
        #[arg(long)]
        draft_model: Option<PathBuf>,

        /// Prompt text
        #[arg(short, long)]
        prompt: String,

        /// Number of tokens to generate
        #[arg(short, long, default_value = "128")]
        max_tokens: usize,

        /// Temperature (0.0 - 2.0)
        #[arg(long, default_value = "0.8")]
        temperature: f32,

        /// Top-k sampling
        #[arg(long, default_value = "40")]
        top_k: i32,

        /// Top-p sampling
        #[arg(long, default_value = "0.95")]
        top_p: f32,

        /// Context size
        #[arg(long, default_value = "2048")]
        context_size: usize,

        // === Speculative Decoding Options ===
        /// Enable speculative decoding
        #[arg(long)]
        speculative: bool,

        /// Draft max tokens (speculation steps, default: 16)
        #[arg(long, default_value = "16")]
        draft_max: usize,

        /// Speculation preset: code, creative, max-speed
        #[arg(long)]
        speculation_preset: Option<String>,

        /// Constrain output to JSON object mode
        #[arg(long)]
        json: bool,
    },

    /// Interactive chat mode
    Chat {
        /// Model path (GGUF file)
        #[arg(short, long)]
        model: PathBuf,

        /// System prompt
        #[arg(long)]
        system_prompt: Option<String>,

        /// Context size
        #[arg(long, default_value = "2048")]
        context_size: usize,
    },

    /// Benchmark model performance
    Benchmark {
        /// Model path (GGUF file)
        #[arg(short, long)]
        model: PathBuf,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,

        /// Prompt length
        #[arg(long, default_value = "512")]
        prompt_length: usize,

        /// Generation length
        #[arg(long, default_value = "128")]
        gen_length: usize,
    },

    /// Model information
    Info {
        /// Model path (GGUF file)
        #[arg(short, long)]
        model: PathBuf,
    },

    /// Convert model to GGUF format
    Convert {
        /// Input model path
        #[arg(short, long)]
        input: PathBuf,

        /// Output GGUF path
        #[arg(short, long)]
        output: PathBuf,

        /// Quantization type
        #[arg(long, default_value = "q4_k")]
        quantize: String,

        /// Output type
        #[arg(long, default_value = "f32")]
        output_type: String,
    },

    /// Server mode
    Server {
        /// Model path (GGUF file)
        #[arg(short, long)]
        model: PathBuf,

        /// Host address
        #[arg(short, long, default_value = "127.0.0.1")]
        host: String,

        /// Port
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },

    /// Show Apple Metal / Apple Silicon capabilities (Phase 7.1)
    MetalInfo {
        /// Also show recommended ContextParams
        #[arg(long)]
        params: bool,
    },

    /// Show WASM/Candle runtime capabilities and recommended config (Phase 7.3)
    WasmInfo {
        /// Model size in billions of parameters (for quant recommendation)
        #[arg(long, default_value = "7.0")]
        model_params_b: f32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Apply performance optimizations first (before any model loading)
    if let Some(preset_str) = &cli.preset {
        use performance::PerformancePreset;

        let preset = match preset_str.as_str() {
            "max-speed" => {
                info!("Applying performance preset: max-speed");
                PerformancePreset::MaxSpeed
            }
            "balanced" => {
                info!("Applying performance preset: balanced");
                PerformancePreset::Balanced
            }
            "max-quality" => {
                info!("Applying performance preset: max-quality");
                PerformancePreset::MaxQuality
            }
            "cpu" => {
                info!("Applying performance preset: cpu");
                PerformancePreset::CPU
            }
            "gpu" => {
                info!("Applying performance preset: gpu");
                PerformancePreset::GPU
            }
            _ => {
                eprintln!(
                    "Unknown preset: {}. Available: max-speed, balanced, max-quality, cpu, gpu",
                    preset_str
                );
                std::process::exit(1);
            }
        };
        preset.apply();
    } else {
        // Apply individual flags if no preset specified
        if cli.cuda_graphs {
            info!("Enabling CUDA Graphs optimization");
            performance::enable_cuda_graphs(true);
        }

        if cli.flash_attn {
            info!("Enabling Flash Attention");
            performance::enable_flash_attention(true);
        }
    }

    // Initialize logging
    let log_level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(log_level).init();

    info!(
        "Barq v{} - High-performance LLM inference engine",
        env!("CARGO_PKG_VERSION")
    );

    // Log performance settings
    if performance::cuda_graphs_enabled() {
        info!("CUDA Graphs: enabled");
    }
    if performance::flash_attention_enabled() {
        info!("Flash Attention: enabled");
    }

    // Execute command
    match cli.command {
        Commands::Run {
            model,
            draft_model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            context_size,
            speculative,
            draft_max,
            speculation_preset,
            json,
        } => {
            cmd_run(
                model,
                draft_model,
                prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                context_size,
                speculative,
                draft_max,
                speculation_preset,
                json,
            )
            .await
        }

        Commands::Chat {
            model,
            system_prompt,
            context_size,
        } => cmd_chat(model, system_prompt, context_size).await,

        Commands::Benchmark {
            model,
            iterations,
            prompt_length,
            gen_length,
        } => cmd_benchmark(model, iterations, prompt_length, gen_length).await,

        Commands::Info { model } => cmd_info(model).await,

        Commands::Convert {
            input,
            output,
            quantize,
            output_type,
        } => cmd_convert(input, output, quantize, output_type).await,

        Commands::Server { model, host, port } => cmd_server(model, host, port).await,

        Commands::MetalInfo { params } => cmd_metal_info(params).await,

        Commands::WasmInfo { model_params_b } => cmd_wasm_info(model_params_b).await,
    }
}

async fn cmd_run(
    model: PathBuf,
    draft_model: Option<PathBuf>,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    context_size: usize,
    speculative: bool,
    draft_max: usize,
    speculation_preset: Option<String>,
    json_mode: bool,
) -> anyhow::Result<()> {
    use models::{
        context::{Batch, ContextParams},
        llama::LlamaModel,
        loader::Model,
    };
    use sampling::{JsonMode, JsonSchema};
    use std::sync::Arc;
    use vocab::{GgufTokenizer, Tokenizer};

    info!("Loading model: {:?}", model);
    info!("Prompt: {}", prompt);
    info!("Max tokens: {}", max_tokens);

    // Load the model
    let loaded_model = Model::load(&model).await?;
    info!("Model loaded successfully");
    info!("Architecture: {:?}", loaded_model.arch());
    info!("Vocab size: {}", loaded_model.hparams().n_vocab);
    info!("Embedding dim: {}", loaded_model.hparams().n_embd);
    info!("Layers: {}", loaded_model.hparams().n_layer);
    let model_vocab_size = loaded_model.hparams().n_vocab as usize;

    // Create tokenizer
    let tokenizer = GgufTokenizer::from_gguf(loaded_model.metadata());

    // Tokenize prompt
    let tokenization_result = tokenizer.tokenize(&prompt, true).await?;
    let prompt_tokens: Vec<i32> = tokenization_result
        .ids
        .iter()
        .map(|&id| id as i32)
        .collect();

    info!("Prompt tokens: {}", prompt_tokens.len());
    if prompt_tokens.is_empty() {
        return Err(anyhow::anyhow!(
            "Prompt tokenized to no tokens; cannot start generation"
        ));
    }
    if json_mode {
        info!("JSON mode: enabled");
    }

    // Create inference context
    let model_arc = Arc::new(loaded_model);
    let llama_model = LlamaModel::new(model_arc.clone())?;
    let params = ContextParams {
        n_ctx: context_size as u32,
        ..Default::default()
    };

    let context = llama_model.create_context(params)?;
    info!("Context created");

    // Generate tokens
    info!("Generating {} tokens...", max_tokens);
    let start = std::time::Instant::now();

    let (generated_tokens, output_text) = if json_mode {
        let schema = JsonSchema::Object {
            properties: vec![],
            required: vec![],
        };
        let grammar_sampler = if tokenizer.vocab().is_empty() {
            JsonMode::create_sampler(&schema, model_vocab_size)?
        } else {
            JsonMode::create_sampler_with_vocab(&schema, tokenizer.vocab())?
        };

        let mut grammar_sampler = grammar_sampler;
        let prompt_batch = Batch::from_tokens(&prompt_tokens);
        let mut logits = context.encode(&prompt_batch).await?;
        let mut generated_tokens = Vec::with_capacity(max_tokens);

        for _ in 0..max_tokens {
            let mut constrained_logits = logits.clone();
            grammar_sampler.apply_constraints(&mut constrained_logits)?;
            let token = context.sample(&constrained_logits, temperature, top_k, top_p)?;
            generated_tokens.push(token);

            let token_text = tokenizer
                .vocab()
                .get_token(token as u32)
                .map(|token| token.text.as_str())
                .unwrap_or("");
            grammar_sampler.accept_token(token as usize, token_text)?;

            if let Some(eos) = tokenizer.vocab().special_tokens().eos {
                if token as u32 == eos {
                    break;
                }
            }

            let batch = Batch::single(token);
            logits = context.decode(&batch).await?;
        }

        if generated_tokens.is_empty() {
            return Err(anyhow::anyhow!("JSON mode produced no tokens"));
        }

        let generated_ids: Vec<u32> = generated_tokens.iter().map(|&id| id as u32).collect();
        let output_text = tokenizer.decode(&generated_ids).await?;
        let parsed_json = JsonMode::validate_output(&output_text)?;
        let pretty_output = serde_json::to_string_pretty(&parsed_json)?;

        (generated_tokens, pretty_output)
    } else {
        let generated_tokens = context
            .generate(&prompt_tokens, max_tokens, temperature, top_k, top_p)
            .await?;
        let generated_ids: Vec<u32> = generated_tokens.iter().map(|&id| id as u32).collect();
        let output_text = tokenizer.decode(&generated_ids).await?;
        (generated_tokens, output_text)
    };

    let elapsed = start.elapsed();
    info!(
        "Generated {} tokens in {:.2}s ({:.2} tokens/s)",
        generated_tokens.len(),
        elapsed.as_secs_f64(),
        generated_tokens.len() as f64 / elapsed.as_secs_f64()
    );

    println!(
        "\n=== {} ===",
        if json_mode { "JSON Output" } else { "Output" }
    );
    println!("{}", output_text);
    println!("\n=== Stats ===");
    println!("Tokens generated: {}", generated_tokens.len());
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Speed: {:.2} tokens/s",
        generated_tokens.len() as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

async fn cmd_chat(
    model: PathBuf,
    system_prompt: Option<String>,
    context_size: usize,
) -> anyhow::Result<()> {
    info!("Starting chat mode with model: {:?}", model);

    if let Some(sp) = &system_prompt {
        info!("System prompt: {}", sp);
    }

    println!("Chat mode starting...");
    println!("Type your message and press Enter to send.");
    println!("Type 'quit' or 'exit' to end the session.\n");

    // TODO: Implement actual chat loop

    Ok(())
}

async fn cmd_benchmark(
    model: PathBuf,
    iterations: usize,
    prompt_length: usize,
    gen_length: usize,
) -> anyhow::Result<()> {
    use benchmark::{BenchmarkConfig, InferenceBenchmark};
    use models::{
        context::{Batch, ContextParams},
        llama::LlamaModel,
        loader::Model,
    };
    use std::sync::Arc;
    use vocab::{GgufTokenizer, Tokenizer};

    info!("Benchmarking model: {:?}", model);
    info!("Iterations: {}", iterations);
    info!("Prompt length: {}", prompt_length);
    info!("Generation length: {}", gen_length);

    let config = BenchmarkConfig {
        runs: iterations,
        warmup_runs: 2,
        prompt_length,
        gen_length,
        measure_ttft: true,
        measure_memory: true,
    };

    let bench = InferenceBenchmark::with_config(config.clone());
    let model_arc = Arc::new(Model::load(&model).await?);
    let llama_model = Arc::new(LlamaModel::new(model_arc.clone())?);
    let tokenizer = Arc::new(GgufTokenizer::from_gguf(model_arc.metadata()));
    let synthetic_prompt = std::iter::repeat("barq")
        .take(prompt_length.max(1))
        .collect::<Vec<_>>()
        .join(" ");
    let context_params = ContextParams::cpu_optimized();
    let temperature = 0.8f32;
    let top_k = 40i32;
    let top_p = 0.95f32;
    let boxed_error = |msg: String| -> Box<dyn std::error::Error> {
        Box::new(std::io::Error::new(std::io::ErrorKind::Other, msg))
    };

    let result = bench.run({
        let llama_model = Arc::clone(&llama_model);
        let tokenizer = Arc::clone(&tokenizer);
        let synthetic_prompt = synthetic_prompt.clone();
        move || {
            tokio::task::block_in_place(|| {
                let handle = tokio::runtime::Handle::current();
                let start = std::time::Instant::now();

                let context = llama_model
                    .create_context(context_params.clone())
                    .map_err(|e| boxed_error(e.to_string()))?;

                let tokenization_result = handle
                    .block_on(tokenizer.tokenize(&synthetic_prompt, true))
                    .map_err(|e| boxed_error(e.to_string()))?;

                let prompt_tokens: Vec<i32> = tokenization_result
                    .ids
                    .iter()
                    .map(|&id| id as i32)
                    .collect();

                if prompt_tokens.is_empty() {
                    return Err(boxed_error(
                        "Synthetic prompt tokenized to no tokens".to_string(),
                    ));
                }

                let prompt_batch = Batch::from_tokens(&prompt_tokens);
                let mut logits = handle
                    .block_on(context.encode(&prompt_batch))
                    .map_err(|e| boxed_error(e.to_string()))?;

                let mut generated_tokens = 0usize;
                let mut ttft = std::time::Duration::ZERO;

                for _ in 0..gen_length {
                    let token = context
                        .sample(&logits, temperature, top_k, top_p)
                        .map_err(|e| boxed_error(e.to_string()))?;

                    generated_tokens += 1;

                    if generated_tokens == 1 {
                        ttft = start.elapsed();
                    }

                    if token == 0 || token == 2 {
                        break;
                    }

                    logits = handle
                        .block_on(context.decode(&Batch::single(token)))
                        .map_err(|e| boxed_error(e.to_string()))?;
                }

                let total_time = start.elapsed();
                Ok(benchmark::BenchmarkSample {
                    total_tokens: prompt_tokens.len() + generated_tokens,
                    total_time,
                    ttft: if ttft.is_zero() { total_time } else { ttft },
                })
            })
        }
    });

    result.print();

    Ok(())
}

async fn cmd_info(model: PathBuf) -> anyhow::Result<()> {
    info!("Getting model info: {:?}", model);
    use models::loader::Model;

    let loaded = Model::load(&model).await?;
    println!("\n=== Model Info ===");
    println!("Architecture: {:?}", loaded.arch());
    println!("Tensor count:  {}", loaded.tensor_count());
    println!("Metadata keys: {}", loaded.metadata().len());
    println!("Context size:  {}", loaded.hparams().n_ctx_train);
    println!("Layers:        {}", loaded.hparams().n_layer);
    println!("Embedding dim: {}", loaded.hparams().n_embd);
    println!("Vocab size:    {}", loaded.hparams().n_vocab);
    println!("==================\n");

    Ok(())
}

async fn cmd_convert(
    input: PathBuf,
    output: PathBuf,
    quantize: String,
    output_type: String,
) -> anyhow::Result<()> {
    info!("Converting model: {:?} -> {:?}", input, output);
    info!("Quantization: {}", quantize);
    info!("Output type: {}", output_type);

    use quant::ik_quant::{quantize_model_ik, repack_model_cpu, IKQuantConfig, IKQuantType};

    let input_str = input.to_string_lossy().to_string();
    let output_str = output.to_string_lossy().to_string();
    let quantize_lower = quantize.to_ascii_lowercase();

    match quantize_lower.as_str() {
        "q4_k_r4" => {
            repack_model_cpu(&input_str, &output_str)?;
        }
        "iq4_ks" => {
            quantize_model_ik(
                &input_str,
                &output_str,
                &IKQuantConfig::new(IKQuantType::IQ4_KS),
            )?;
        }
        "iq3_ks" => {
            quantize_model_ik(
                &input_str,
                &output_str,
                &IKQuantConfig::new(IKQuantType::IQ3_KS),
            )?;
        }
        "iq2_ks" => {
            quantize_model_ik(
                &input_str,
                &output_str,
                &IKQuantConfig::new(IKQuantType::IQ2_KS),
            )?;
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported quantization type '{}'. Supported: q4_k_r4, iq4_ks, iq3_ks, iq2_ks",
                other
            ));
        }
    }

    Ok(())
}

async fn cmd_server(model: PathBuf, host: String, port: u16) -> anyhow::Result<()> {
    info!("Starting server on {}:{}", host, port);
    info!("Model: {:?}", model);

    use advanced::uds_server::{
        InferenceHandler, InferenceRequest, InferenceResponse, InferenceServer, ServerConfig,
    };
    use async_trait::async_trait;
    use barq_core::error::Result;
    use models::{context::ContextParams, llama::LlamaModel, loader::Model};
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use vocab::{GgufTokenizer, Tokenizer};

    // Load actual model
    info!("Loading model for UDS Server: {:?}", model);
    let loaded_model = Model::load(&model).await?;
    let model_arc = Arc::new(loaded_model);
    let llama_model = Arc::new(LlamaModel::new(model_arc.clone())?);
    let tokenizer = Arc::new(GgufTokenizer::from_gguf(model_arc.metadata()));

    struct ServerHandler {
        model: Arc<LlamaModel>,
        tokenizer: Arc<GgufTokenizer>,
    }

    #[async_trait]
    impl InferenceHandler for ServerHandler {
        async fn process_request(
            &self,
            request: InferenceRequest,
            response_tx: mpsc::Sender<Result<InferenceResponse>>,
        ) {
            let start = std::time::Instant::now();
            let params = ContextParams::default();

            let context = match self.model.create_context(params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = response_tx.send(Err(e)).await;
                    return;
                }
            };

            let tokenization_result = match self.tokenizer.tokenize(&request.prompt, true).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = response_tx
                        .send(Err(barq_core::error::Error::Backend(e.to_string())))
                        .await;
                    return;
                }
            };

            let mut current_tokens: Vec<i32> = tokenization_result
                .ids
                .into_iter()
                .map(|id| id as i32)
                .collect();

            let batch = models::context::Batch::from_tokens(&current_tokens);
            let mut logits = match context.encode(&batch).await {
                Ok(l) => l,
                Err(e) => {
                    let _ = response_tx.send(Err(e)).await;
                    return;
                }
            };

            let ttft = start.elapsed().as_millis() as u64;

            for i in 0..request.max_tokens {
                let token = match context.sample(&logits, request.temperature, 40, 0.95) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = response_tx.send(Err(e)).await;
                        break;
                    }
                };

                let decode_res = context.decode(&models::context::Batch::single(token)).await;

                let decoded_text = self
                    .tokenizer
                    .decode(&[token as u32])
                    .await
                    .unwrap_or_default();
                let done = token == 0 || token == 2 || i == request.max_tokens - 1;

                let response = InferenceResponse {
                    id: request.id.clone(),
                    text: decoded_text,
                    tokens_generated: i + 1,
                    ttft_ms: ttft,
                    total_time_ms: start.elapsed().as_millis() as u64,
                    done,
                };

                if response_tx.send(Ok(response)).await.is_err() {
                    break;
                }

                if done {
                    break;
                }

                current_tokens.push(token);

                match decode_res {
                    Ok(l) => logits = l,
                    Err(e) => {
                        let _ = response_tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        }
    }

    let handler = Arc::new(ServerHandler {
        model: llama_model,
        tokenizer,
    });

    let config = ServerConfig {
        socket_path: std::path::PathBuf::from(format!("/tmp/barq-inference-{}.sock", port)),
        ..Default::default()
    };

    let mut server = InferenceServer::with_handler(config, handler);
    info!(
        "Starting async UDS streaming server. TCP configuration ({}:{}) ignored.",
        host, port
    );

    server.start().await?;

    Ok(())
}

// ── Phase 6.2: Continuous BatchEngine handler ─────────────────────────────────

async fn cmd_server_batch(model: PathBuf, host: String, port: u16) -> anyhow::Result<()> {
    info!("Starting continuous-batching server on {}:{}", host, port);

    use advanced::uds_server::{
        InferenceHandler, InferenceRequest, InferenceResponse, InferenceServer, ServerConfig,
    };
    use advanced::{BatchEngine, BatchEngineHandle, ContinuousBatchingConfig};
    use async_trait::async_trait;
    use barq_core::error::Result;
    use models::{llama::LlamaModel, loader::Model};
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use vocab::{GgufTokenizer, Tokenizer};

    info!("Loading model for continuous-batching server: {:?}", model);
    let loaded_model = Model::load(&model).await?;
    let model_arc = Arc::new(loaded_model);
    let llama_model = Arc::new(LlamaModel::new(model_arc.clone())?);
    let tokenizer = Arc::new(GgufTokenizer::from_gguf(model_arc.metadata()));

    // Build the BatchEngine (spawned as a background task)
    let transformer = llama_model.transformer();
    let batch_config = ContinuousBatchingConfig::default();
    let (batch_engine, engine_handle) =
        BatchEngine::new(model_arc.clone(), transformer, batch_config);

    tokio::spawn(async move {
        batch_engine.run().await;
    });

    struct BatchHandler {
        engine: BatchEngineHandle,
        tokenizer: Arc<GgufTokenizer>,
    }

    #[async_trait]
    impl InferenceHandler for BatchHandler {
        async fn process_request(
            &self,
            request: InferenceRequest,
            response_tx: mpsc::Sender<Result<InferenceResponse>>,
        ) {
            let start = std::time::Instant::now();

            let tokenization_result = match self.tokenizer.tokenize(&request.prompt, true).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = response_tx
                        .send(Err(barq_core::error::Error::Backend(e.to_string())))
                        .await;
                    return;
                }
            };

            let tokens: Vec<i32> = tokenization_result
                .ids
                .into_iter()
                .map(|id| id as i32)
                .collect();

            let ttft = start.elapsed().as_millis() as u64;

            // Submit to the shared batch engine
            let mut rx = match self.engine.submit(tokens, request.max_tokens).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = response_tx.send(Err(e)).await;
                    return;
                }
            };

            let mut tokens_generated = 0u32;
            while let Some(token) = rx.recv().await {
                tokens_generated += 1;
                let done = token == 0 || token == 2;

                let decoded_text = self
                    .tokenizer
                    .decode(&[token as u32])
                    .await
                    .unwrap_or_default();

                let response = InferenceResponse {
                    id: request.id.clone(),
                    text: decoded_text,
                    tokens_generated: tokens_generated as usize,
                    ttft_ms: ttft,
                    total_time_ms: start.elapsed().as_millis() as u64,
                    done,
                };

                if response_tx.send(Ok(response)).await.is_err() || done {
                    break;
                }
            }
        }
    }

    let handler = Arc::new(BatchHandler {
        engine: engine_handle,
        tokenizer,
    });

    let config = ServerConfig {
        socket_path: std::path::PathBuf::from(format!("/tmp/barq-inference-batch-{}.sock", port)),
        ..Default::default()
    };

    let mut server = InferenceServer::with_handler(config, handler);
    info!("Continuous batching server ready.");
    server.start().await?;

    Ok(())
}

// ── Phase 7.1/7.2: Metal info command ────────────────────────────────────────

async fn cmd_metal_info(show_params: bool) -> anyhow::Result<()> {
    use advanced::metal_backend_integration::{apply_metal_optimizations, print_report};
    use models::context::ContextParams;

    println!("Detecting Apple Metal / Silicon capabilities...\n");

    let report = apply_metal_optimizations(ContextParams::default());
    print_report(&report);

    if show_params {
        let p = &report.params;
        println!("Recommended ContextParams:");
        println!("  n_ctx:       {}", p.n_ctx);
        println!("  n_threads:   {}", p.n_threads);
        println!("  n_gpu_layers: {}", p.n_gpu_layers);
        println!("  flash_attn:  {}", p.flash_attn);
    }

    println!("Inference score: {}/100", report.caps.inference_score());
    if report.caps.should_offload_all_layers() {
        println!("  ✓ Sufficient memory for full-GPU offload of 7B models");
    }

    Ok(())
}

// ── Phase 7.3: WASM info command ──────────────────────────────────────────────

async fn cmd_wasm_info(model_params_b: f32) -> anyhow::Result<()> {
    use advanced::wasm_candle::{best_wasm_quant, WasmRuntime};

    println!("Detecting WASM/Candle runtime capabilities...\n");

    let rt = WasmRuntime::auto();
    rt.print_summary();

    let caps = &rt.caps;
    println!("Environment:");
    println!("  SharedArrayBuffer: {}", caps.shared_array_buffer);
    println!("  WASM SIMD:         {}", caps.wasm_simd);
    println!("  WebGPU:            {}", caps.webgpu);

    let heap_gb = if caps.usable_heap_bytes == usize::MAX {
        16.0f32 // native — assume generous
    } else {
        caps.usable_heap_bytes as f32 / (1u64 << 30) as f32
    };

    let best_quant = best_wasm_quant(model_params_b, heap_gb);
    println!(
        "\nFor a {:.1}B parameter model on {:.1} GB heap:",
        model_params_b, heap_gb
    );
    println!("  Recommended quantization: {:?}", best_quant);
    println!("  Bits per weight: {:.2}", best_quant.bits_per_weight());
    println!("  Browser safe: {}", best_quant.browser_safe());

    Ok(())
}
