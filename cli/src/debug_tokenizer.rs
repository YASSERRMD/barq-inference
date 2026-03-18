use barq_core::loader::Model;
use vocab::{GgufTokenizer, Tokenizer};

#[tokio::main]
async fn main() {
    let model = Model::load(std::path::Path::new("../tmp/models/tinyllama-q4_k_m.gguf")).await.unwrap();
    let tokenizer = GgufTokenizer::from_gguf(model.metadata());
    let prompt = "Hello, my name is";
    let res = tokenizer.tokenize(prompt, true).await.unwrap();
    println!("Token IDs: {:?}", res.ids);
    for id in res.ids {
        println!("Token {}: {:?}", id, tokenizer.decode(&[id]).await.unwrap());
    }
}
