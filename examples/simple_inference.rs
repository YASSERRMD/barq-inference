//! Simple inference example

use models::loader::{Model, ModelLoader};
use models::context::{ModelContext, ContextParams};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model
    let model = Model::load("path/to/model.gguf").await?;
    let model = Arc::new(model);

    // Create context
    let params = ContextParams {
        n_ctx: 2048,
        n_threads: 4,
        ..Default::default()
    };

    let ctx = ModelContext::new(model, params)?;

    // Tokenize prompt (placeholder)
    let tokens = vec![1, 2, 3, 4, 5];

    // Generate text
    let output = ctx.generate(&tokens, 100).await?;

    println!("Generated {} tokens", output.len());

    Ok(())
}
