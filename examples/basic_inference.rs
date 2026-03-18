//! Basic inference example
//!
//! Demonstrates how to:
//! - Load a GGUF model
//! - Create an inference context
//! - Generate text with sampling

use barq_inference::client::{Client, GenerationConfig, SamplingConfig};
use barq_inference::models::ModelLoader;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("=== Barq Inference - Basic Example ===\n");

    // 1. Load a model
    println!("Loading model...");
    let model_path = "path/to/model.gguf"; // Replace with actual path

    let loader = ModelLoader::new();
    let model = loader.load_from_file(model_path).await?;

    println!("Model loaded successfully!");
    println!("  Architecture: {:?}", model.architecture());
    println!("  Parameters: {}", model.parameter_count());
    println!("  Quantization: {:?}\n", model.quantization());

    // 2. Create inference client
    let client = Client::new(model)?;

    // 3. Configure generation
    let config = GenerationConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        ..Default::default()
    };

    // 4. Generate text
    let prompt = "The future of artificial intelligence is";

    println!("Generating text...");
    println!("Prompt: {}\n", prompt);
    println!("Response: ");

    let mut stream = client.generate_stream(prompt, &config).await?;

    while let Some(token) = stream.next().await? {
        print!("{}", token.text);
        std::io::Write::flush(&mut std::io::stdout())?;
    }

    println!("\n\nGeneration complete!");

    Ok(())
}
