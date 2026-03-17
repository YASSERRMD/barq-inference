//! BPE tokenizer implementation

use crate::tokenizer::Tokenizer;
use crate::vocab::{TokenizationResult, Vocab};
use anyhow::Result;
use async_trait::async_trait;

/// BPE tokenizer (GPT-2, GPT-3)
pub struct BpeTokenizer {
    vocab: Vocab,
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: Vocab::new(crate::vocab::VocabType::BPE),
        }
    }
}

#[async_trait]
impl Tokenizer for BpeTokenizer {
    async fn tokenize(&self, _text: &str, _add_special: bool) -> Result<TokenizationResult> {
        Err(anyhow::anyhow!("BPE not yet implemented"))
    }

    async fn decode(&self, _ids: &[u32]) -> Result<String> {
        Err(anyhow::anyhow!("BPE not yet implemented"))
    }

    fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    fn tokenizer_type(&self) -> crate::tokenizer::TokenizerType {
        crate::tokenizer::TokenizerType::BPE
    }
}
