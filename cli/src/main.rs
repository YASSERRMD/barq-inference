//! Barq - High-performance LLM inference engine CLI

mod performance;
mod benchmark;

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
            ).await
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
) -> anyhow::Result<()> {
    info!("Loading model: {:?}", model);
    info!("Prompt: {}", prompt);
    info!("Max tokens: {}", max_tokens);

    // Determine if speculative decoding should be used
    let use_speculative = speculative || draft_model.is_some();

    if use_speculative {
        info!("=== Speculative Decoding Mode ===");

        // Determine draft model path
        let draft_path = if let Some(draft) = draft_model {
            draft
        } else {
            // Auto-select draft model
            info!("Auto-selecting draft model based on target model...");
            model.clone()
        };

        info!("Draft model: {:?}", draft_path);
        info!("Draft max tokens: {}", draft_max);

        if let Some(preset) = &speculation_preset {
            info!("Speculation preset: {}", preset);
        }

        // TODO: Implement actual speculative decoding
        // For now, just show what would be done
        println!("\n[Speculative Decoding]");
        println!("Target model: {:?}", model);
        println!("Draft model: {:?}", draft_path);
        println!("Draft max: {}", draft_max);
        println!("\nTODO: Integrate with SpeculativeEngine");
        println!("Prompt: {}", prompt);

    } else {
        info!("=== Standard Inference Mode ===");

        // TODO: Implement actual inference
        println!("Running inference...");
        println!("Prompt: {}", prompt);
    }

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

    // TODO: Implement actual inference function
    // For now, use a mock that simulates inference
    let result = bench.run(|| {
        // Simulate inference time
        let simulated_time = std::time::Duration::from_millis(100);
        std::thread::sleep(simulated_time);

        let total_tokens = config.prompt_length + config.gen_length;
        Ok((total_tokens, simulated_time))
    });

    result.print();

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
