//! SentencePiece tokenizer implementation

use crate::tokenizer::Tokenizer;
use crate::vocab::{TokenizationResult, Vocab};
use anyhow::Result;
use async_trait::async_trait;

/// SentencePiece tokenizer (LLaMA, Mistral, etc.)
pub struct SentencePiece {
    vocab: Vocab,
}

impl Default for SentencePiece {
    fn default() -> Self {
        Self::new()
    }
}

impl SentencePiece {
    /// Create a new SentencePiece tokenizer
    pub fn new() -> Self {
        Self {
            vocab: Vocab::new(crate::vocab::VocabType::SPM),
        }
    }
}

#[async_trait]
impl Tokenizer for SentencePiece {
    async fn tokenize(&self, _text: &str, _add_special: bool) -> Result<TokenizationResult> {
        Err(anyhow::anyhow!("SentencePiece not yet implemented"))
    }

    async fn decode(&self, _ids: &[u32]) -> Result<String> {
        Err(anyhow::anyhow!("SentencePiece not yet implemented"))
    }

    fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    fn tokenizer_type(&self) -> crate::tokenizer::TokenizerType {
        crate::tokenizer::TokenizerType::SPM
    }
}
