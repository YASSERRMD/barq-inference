//! Unigram tokenizer implementation

use crate::tokenizer::Tokenizer;
use crate::vocab::{TokenizationResult, Vocab};
use anyhow::Result;
use async_trait::async_trait;

/// Unigram tokenizer (T5)
pub struct Unigram {
    vocab: Vocab,
}

impl Default for Unigram {
    fn default() -> Self {
        Self::new()
    }
}

impl Unigram {
    pub fn new() -> Self {
        Self {
            vocab: Vocab::new(crate::vocab::VocabType::UGM),
        }
    }
}

#[async_trait]
impl Tokenizer for Unigram {
    async fn tokenize(&self, _text: &str, _add_special: bool) -> Result<TokenizationResult> {
        Err(anyhow::anyhow!("Unigram not yet implemented".to_string()))
    }

    async fn decode(&self, _ids: &[u32]) -> Result<String> {
        Err(anyhow::anyhow!("Unigram not yet implemented".to_string()))
    }

    fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    fn tokenizer_type(&self) -> crate::tokenizer::TokenizerType {
        crate::tokenizer::TokenizerType::UGM
    }
}
