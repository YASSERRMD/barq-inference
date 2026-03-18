//! Advanced inference example
//!
//! Demonstrates advanced features:
//! - Grammar-guided generation (JSON mode)
//! - Custom sampling strategies
//! - Batch processing
//! - Streaming with callbacks

use barq_inference::client::{Client, GenerationConfig, SamplingStrategy, TokenCallback};
use barq_inference::grammar::GrammarCompiler;
use barq_inference::sampling::{Temperature, TopK, TopP, SamplerChain};
use barq_inference::models::ModelLoader;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Barq Inference - Advanced Example ===\n");

    let loader = ModelLoader::new();
    let model_path = "path/to/model.gguf";
    let model = loader.load_from_file(model_path).await?;
    let client = Client::new(model)?;

    // ========================================================================
    // Example 1: JSON Mode with Grammar
    // ========================================================================

    println!("1. JSON Mode Example\n");

    let json_schema = r#"
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "email": {"type": "string"}
  },
  "required": ["name", "age", "email"]
}
"#;

    let compiler = GrammarCompiler::new();
    let grammar = compiler.compile_json(json_schema)?;

    let config = GenerationConfig {
        max_tokens: 200,
        grammar: Some(grammar),
        ..Default::default()
    };

    let prompt = "Generate a user profile in JSON format:";
    let response = client.generate(prompt, &config).await?;

    println!("Prompt: {}\n", prompt);
    println!("Response:\n{}\n", response.text);

    // ========================================================================
    // Example 2: Custom Sampling Chain
    // ========================================================================

    println!("2. Custom Sampling Chain\n");

    // Build custom sampling pipeline
    let mut sampler = SamplerChain::new();

    // Apply repetition penalty first
    sampler.add(Box::new(barq_inference::sampling::RepetitionPenalty::new(1.1)));

    // Then temperature
    sampler.add(Box::new(Temperature::new(0.8)));

    // Then top-k
    sampler.add(Box::new(TopK::new(40)));

    // Finally top-p
    sampler.add(Box::new(TopP::new(0.95)));

    let custom_config = GenerationConfig {
        max_tokens: 100,
        sampler: Some(sampler),
        ..Default::default()
    };

    let prompt = "Write a haiku about programming:";
    let response = client.generate(prompt, &custom_config).await?;

    println!("Prompt: {}\n", prompt);
    println!("Response:\n{}\n", response.text);

    // ========================================================================
    // Example 3: Batch Processing
    // ========================================================================

    println!("3. Batch Processing\n");

    let prompts = vec![
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Brazil?",
    ];

    let batch_config = GenerationConfig {
        max_tokens: 50,
        temperature: 0.3, // Lower temperature for factual responses
        ..Default::default()
    };

    let responses = client.generate_batch(prompts, &batch_config).await?;

    for (i, response) in responses.iter().enumerate() {
        println!("Prompt {}: {}", i + 1, response.prompt);
        println!("Response {}: {}\n", i + 1, response.text.trim());
    }

    // ========================================================================
    // Example 4: Streaming with Callback
    // ========================================================================

    println!("4. Streaming with Callback\n");

    // Shared state to collect tokens
    let tokens = Arc::new(RwLock::new(Vec::new()));
    let tokens_clone = tokens.clone();

    // Create callback
    let callback = TokenCallback::new(move |token| {
        let mut tokens = tokens_clone.write().unwrap();
        tokens.push(token.text.clone());
        println!("{}", token.text);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        true // Return true to continue generation
    });

    let stream_config = GenerationConfig {
        max_tokens: 150,
        callback: Some(callback),
        ..Default::default()
    };

    let prompt = "Write a short story about a robot learning to paint:";
    println!("Prompt: {}\n", prompt);
    println!("Response: ");

    client.generate(prompt, &stream_config).await?;

    println!("\n\nTotal tokens: {}", tokens.read().unwrap().len());

    // ========================================================================
    // Example 5: Multi-GPU Inference (if available)
    // ========================================================================

    #[cfg(feature = "cuda")]
    {
        println!("5. Multi-GPU Inference\n");

        // Use tensor parallelism across 2 GPUs
        let gpu_config = barq_inference::backend::MultiGpuConfig::tensor_parallel(2);
        let multi_gpu_client = Client::with_multi_gpu(model, gpu_config)?;

        let prompt = "Explain quantum computing in simple terms:";
        let config = GenerationConfig {
            max_tokens: 100,
            ..Default::default()
        };

        let response = multi_gpu_client.generate(prompt, &config).await?;
        println!("Response: {}\n", response.text.trim());
    }

    Ok(())
}
