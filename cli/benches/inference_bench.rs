use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use models::{context::ContextParams, llama::LlamaModel, loader::Model};
use std::sync::Arc;
use tokio::runtime::Runtime;
use vocab::{ChatMessage, ChatRole, ChatTemplate, GgufTokenizer, Tokenizer};

#[path = "../tests/integration/fixtures.rs"]
mod fixtures;

fn bench_prompt_tokenization(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let path = fixtures::tiny_llama_fixture("bench-tokenization");
    let model = rt.block_on(async { Model::load(&path).await.expect("model should load") });
    let tokenizer = GgufTokenizer::from_gguf(model.metadata());
    let prompt = "hello world";

    c.bench_function("prompt_tokenization", |b| {
        b.iter(|| {
            rt.block_on(async {
                let result = tokenizer.tokenize(black_box(prompt), true).await;
                black_box(result.expect("tokenization should succeed"));
            });
        })
    });
}

fn bench_chat_prompt_rendering(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let path = fixtures::tiny_llama_fixture("bench-rendering");
    let model = rt.block_on(async { Model::load(&path).await.expect("model should load") });
    let tokenizer = GgufTokenizer::from_gguf(model.metadata());
    let template = ChatTemplate::for_arch(model.arch().name());
    let messages = vec![
        ChatMessage::new(ChatRole::User, "Summarize the design"),
        ChatMessage::new(ChatRole::Assistant, "Barq is a Rust LLM engine."),
    ];

    c.bench_function("chat_prompt_rendering", |b| {
        b.iter(|| {
            let rendered = template
                .render(Some(tokenizer.vocab()), Some("You are concise"), &messages)
                .expect("template render should succeed");
            black_box(rendered);
        })
    });
}

fn bench_token_generation(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let path = fixtures::tiny_llama_fixture("bench-generation");
    let model = rt.block_on(async { Model::load(&path).await.expect("model should load") });
    let tokenizer = GgufTokenizer::from_gguf(model.metadata());
    let model = Arc::new(model);
    let llama = Arc::new(LlamaModel::new(Arc::clone(&model)).expect("llama wrapper"));
    let context_params = ContextParams::cpu_optimized();
    let prompt_tokens: Vec<i32> = rt
        .block_on(async {
            tokenizer
                .tokenize("hello", true)
                .await
                .expect("tokenization")
        })
        .ids
        .into_iter()
        .map(|id| id as i32)
        .collect();

    c.bench_function("token_generation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let context = llama
                    .create_context(context_params.clone())
                    .expect("context should build");
                let generated = context
                    .generate(&prompt_tokens, 1, 0.0, 1, 1.0)
                    .await
                    .expect("generation should succeed");
                black_box(generated);
            });
        })
    });
}

fn bench_memory_allocation(c: &mut Criterion) {
    use barq_core::memory::{Allocator, DefaultAllocator};

    let allocator = DefaultAllocator::new(8 * 1024 * 1024).expect("allocator");
    let mut group = c.benchmark_group("memory_allocation");

    for size in [1024usize, 4096, 16384, 65536] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let buffer = allocator.allocate(size).expect("allocation");
                black_box(buffer.as_ptr());
                black_box(buffer.size());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_prompt_tokenization,
    bench_chat_prompt_rendering,
    bench_token_generation,
    bench_memory_allocation
);
criterion_main!(benches);
