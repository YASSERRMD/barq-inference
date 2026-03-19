use super::fixtures::tiny_llama_fixture;
use models::{context::ContextParams, llama::LlamaModel, loader::Model};
use std::sync::Arc;
use vocab::{GgufTokenizer, Tokenizer};

#[tokio::test]
async fn generates_expected_token_from_tiny_fixture() {
    let path = tiny_llama_fixture("end-to-end");
    let model = Arc::new(Model::load(&path).await.expect("model should load"));
    let tokenizer = GgufTokenizer::from_gguf(model.metadata());
    let llama = LlamaModel::new(Arc::clone(&model)).expect("llama wrapper should build");
    let context = llama
        .create_context(ContextParams::cpu_optimized())
        .expect("context should build");

    let prompt_tokens: Vec<i32> = tokenizer
        .tokenize("hello", true)
        .await
        .expect("tokenization should succeed")
        .ids
        .into_iter()
        .map(|id| id as i32)
        .collect();

    let generated = context
        .generate(&prompt_tokens, 1, 0.0, 1, 1.0)
        .await
        .expect("generation should succeed");

    assert_eq!(generated, vec![3]);

    let generated_ids: Vec<u32> = generated.iter().map(|&id| id as u32).collect();
    let decoded = tokenizer
        .decode(&generated_ids)
        .await
        .expect("decode should succeed");

    assert_eq!(decoded, "world");
}
