use super::fixtures::tiny_llama_fixture;
use models::{loader::Model, LlmArch};
use vocab::{GgufTokenizer, Tokenizer};

#[tokio::test]
async fn loads_minimal_llama_model() {
    let path = tiny_llama_fixture("model-loading");

    let model = Model::load(&path).await.expect("model should load");
    assert_eq!(model.arch(), LlmArch::Llama);
    assert_eq!(model.tensor_count(), 2);
    assert_eq!(model.hparams().n_layer, 0);
    assert_eq!(model.hparams().n_vocab, 128);

    let tokenizer = GgufTokenizer::from_gguf(model.metadata());
    let tokenized = tokenizer
        .tokenize("hello", true)
        .await
        .expect("tokenization should succeed");

    assert!(!tokenized.ids.is_empty());
}
