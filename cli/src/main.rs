//! Barq - High-performance LLM inference engine CLI

mod performance;

use std::path::PathBuf;
use clap::{Parser, Subcommand};
use tracing::{info, error, Level};

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
                eprintln!("Unknown preset: {}. Available: max-speed, balanced, max-quality, cpu, gpu", preset_str);
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
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .init();

    info!("Barq v{} - High-performance LLM inference engine", env!("CARGO_PKG_VERSION"));

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
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            context_size,
        } => {
            cmd_run(model, prompt, max_tokens, temperature, top_k, top_p, context_size).await
        }

        Commands::Chat {
            model,
            system_prompt,
            context_size,
        } => {
            cmd_chat(model, system_prompt, context_size).await
        }

        Commands::Benchmark {
            model,
            iterations,
            prompt_length,
            gen_length,
        } => {
            cmd_benchmark(model, iterations, prompt_length, gen_length).await
        }

        Commands::Info { model } => {
            cmd_info(model).await
        }

        Commands::Convert {
            input,
            output,
            quantize,
            output_type,
        } => {
            cmd_convert(input, output, quantize, output_type).await
        }

        Commands::Server {
            model,
            host,
            port,
        } => {
            cmd_server(model, host, port).await
        }
    }
}

async fn cmd_run(
    model: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    context_size: usize,
) -> anyhow::Result<()> {
    info!("Loading model: {:?}", model);
    info!("Prompt: {}", prompt);
    info!("Max tokens: {}", max_tokens);

    // TODO: Implement actual inference
    println!("Running inference...");
    println!("Prompt: {}", prompt);

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
    info!("Benchmarking model: {:?}", model);
    info!("Iterations: {}", iterations);
    info!("Prompt length: {}", prompt_length);
    info!("Generation length: {}", gen_length);

    // TODO: Implement actual benchmarking

    Ok(())
}

async fn cmd_info(model: PathBuf) -> anyhow::Result<()> {
    info!("Getting model info: {:?}", model);

    // TODO: Implement actual model info extraction

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

    // TODO: Implement actual model conversion

    Ok(())
}

async fn cmd_server(model: PathBuf, host: String, port: u16) -> anyhow::Result<()> {
    info!("Starting server on {}:{}", host, port);
    info!("Model: {:?}", model);

    // TODO: Implement actual server

    Ok(())
}
