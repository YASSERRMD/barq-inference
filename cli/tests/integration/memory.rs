use super::fixtures::tiny_llama_fixture;
use models::{context::ContextParams, llama::LlamaModel, loader::Model};
use std::sync::Arc;

#[tokio::test]
async fn repeated_model_initialization_smoke_test() {
    let path = tiny_llama_fixture("memory");

    for _ in 0..8 {
        let model = Arc::new(Model::load(&path).await.expect("model should load"));
        assert_eq!(model.tensor_count(), 2);

        let llama = LlamaModel::new(Arc::clone(&model)).expect("llama wrapper should build");
        let _context = llama
            .create_context(ContextParams::cpu_optimized())
            .expect("context should build");
    }
}
