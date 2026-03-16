//! SentencePiece tokenizer implementation

use crate::tokenizer::Tokenizer;
use crate::vocab::{TokenizationResult, Vocab};
use async_trait::async_trait;
use core::error::Error;

/// SentencePiece tokenizer (LLaMA, Mistral, etc.)
pub struct SentencePiece {
    vocab: Vocab,
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
    async fn tokenize(&self, _text: &str, _add_special: bool) -> Result<TokenizationResult, Error> {
        Err(Error::Unsupported("SentencePiece not yet implemented".to_string()))
    }

    async fn decode(&self, _ids: &[u32]) -> Result<String, Error> {
        Err(Error::Unsupported("SentencePiece not yet implemented".to_string()))
    }

    fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    fn tokenizer_type(&self) -> crate::tokenizer::TokenizerType {
        crate::tokenizer::TokenizerType::SPM
    }
}
