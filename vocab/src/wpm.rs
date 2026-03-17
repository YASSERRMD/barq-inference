//! WordPiece tokenizer implementation

use crate::tokenizer::Tokenizer;
use crate::vocab::{TokenizationResult, Vocab};
use anyhow::Result;
use async_trait::async_trait;

/// WordPiece tokenizer (BERT)
pub struct WordPiece {
    vocab: Vocab,
}

impl Default for WordPiece {
    fn default() -> Self {
        Self::new()
    }
}

impl WordPiece {
    pub fn new() -> Self {
        Self {
            vocab: Vocab::new(crate::vocab::VocabType::WPM),
        }
    }
}

#[async_trait]
impl Tokenizer for WordPiece {
    async fn tokenize(&self, _text: &str, _add_special: bool) -> Result<TokenizationResult> {
        Err(anyhow::anyhow!("WordPiece not yet implemented".to_string()))
    }

    async fn decode(&self, _ids: &[u32]) -> Result<String> {
        Err(anyhow::anyhow!("WordPiece not yet implemented".to_string()))
    }

    fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    fn tokenizer_type(&self) -> crate::tokenizer::TokenizerType {
        crate::tokenizer::TokenizerType::WPM
    }
}
