//! Tokenizer trait and common implementations

use async_trait::async_trait;

use crate::vocab::{TokenizationResult, Vocab, VocabType};
use anyhow::Result;

/// Tokenizer trait
#[async_trait]
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into token IDs
    async fn tokenize(&self, text: &str, add_special: bool) -> Result<TokenizationResult>;

    /// Decode token IDs back to text
    async fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Returns the vocabulary
    fn vocab(&self) -> &Vocab;

    /// Returns the tokenizer type
    fn tokenizer_type(&self) -> TokenizerType;
}

/// Tokenizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenizerType {
    /// SentencePiece (LLaMA, Mistral, etc.)
    SPM,
    /// Byte-Pair Encoding (GPT-2, GPT-3)
    BPE,
    /// WordPiece (BERT)
    WPM,
    /// Unigram (T5)
    UGM,
    /// RWKV tokenizer
    RWKV,
    /// PLaMo-2 tokenizer
    PLaMo2,
}

/// Simple whitespace tokenizer for testing
pub struct WhitespaceTokenizer {
    vocab: Vocab,
}

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer
    pub fn new() -> Self {
        let mut vocab = Vocab::new(VocabType::SPM);
        vocab.add_special_tokens(crate::vocab::SpecialToken {
            bos: Some(0),
            eos: Some(1),
            eot: None,
            sep: None,
            nl: None,
            pad: None,
            mask: None,
            fim_pre: None,
            fim_suf: None,
            fim_mid: None,
            fim_pad: None,
        });

        Self { vocab }
    }
}

#[async_trait]
impl Tokenizer for WhitespaceTokenizer {
    async fn tokenize(&self, text: &str, add_special: bool) -> Result<TokenizationResult, Error> {
        let mut tokens = Vec::new();
        let mut ids = Vec::new();

        if add_special {
            if let Some(bos) = self.vocab.special_tokens().bos {
                ids.push(bos);
                tokens.push("<bos>".to_string());
            }
        }

        for word in text.split_whitespace() {
            // Simple hash-based token ID
            let id = (word.len() % 1000) as u32 + 2;
            ids.push(id);
            tokens.push(word.to_string());
        }

        if add_special {
            if let Some(eos) = self.vocab.special_tokens().eos {
                ids.push(eos);
                tokens.push("<eos>".to_string());
            }
        }

        Ok(TokenizationResult::new(ids, tokens))
    }

    async fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let words: Vec<&str> = ids.iter().filter_map(|&id| {
            match id {
                0 => Some("<bos>"),
                1 => Some("<eos>"),
                _ => Some("word"),
            }
        }).collect();

        Ok(words.join(" "))
    }

    fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    fn tokenizer_type(&self) -> TokenizerType {
        TokenizerType::SPM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let result = tokenizer.tokenize("hello world", false).await.unwrap();

        assert_eq!(result.ids.len(), 2);
        assert_eq!(result.tokens, vec!["hello", "world"]);
    }
}
