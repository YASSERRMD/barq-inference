//! GGUF tokenizer implementation

use std::collections::HashMap;
use std::sync::Arc;

use crate::tokenizer::{Tokenizer, TokenizerType};
use crate::vocab::{TokenizationResult, Vocab, VocabType};
use anyhow::Result;

/// Simple GGUF tokenizer
pub struct GgufTokenizer {
    vocab: Arc<Vocab>,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
}

impl GgufTokenizer {
    /// Create a new GGUF tokenizer
    pub fn new() -> Self {
        let vocab = Arc::new(Vocab::new(VocabType::SPM));
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Add basic tokens (placeholder)
        token_to_id.insert("<s>".to_string(), 0);
        token_to_id.insert("</s>".to_string(), 1);
        id_to_token.insert(0, "<s>".to_string());
        id_to_token.insert(1, "</s>".to_string());

        Self {
            vocab,
            token_to_id,
            id_to_token,
        }
    }

    /// Load tokenizer from GGUF model metadata
    pub fn from_gguf(tokens: &HashMap<String, String>) -> Self {
        let vocab = Arc::new(Vocab::new(VocabType::SPM));
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Try to extract tokenizer information from GGUF
        for (key, value) in tokens {
            if key.starts_with("tokenizer.ggml.") {
                // GGUF format tokens
                if let Some(id_str) = key.strip_prefix("tokenizer.ggml.token.") {
                    if let Ok(id) = id_str.parse::<u32>() {
                        id_to_token.insert(id, value.clone());
                        token_to_id.insert(value.clone(), id);
                    }
                }
            }
        }

        Self {
            vocab,
            token_to_id,
            id_to_token,
        }
    }

    /// Simple byte-level tokenization
    fn tokenize_bytes(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Start with BOS token
        if let Some(&bos) = self.vocab.special_tokens().bos.as_ref() {
            tokens.push(bos);
        }

        // Simple character/byte level encoding
        for byte in text.bytes() {
            // Map bytes to token IDs (simple approach)
            let token_id = 2 + (byte as u32); // Offset by special tokens
            tokens.push(token_id);
        }

        // End with EOS token
        if let Some(&eos) = self.vocab.special_tokens().eos.as_ref() {
            tokens.push(eos);
        }

        tokens
    }

    /// Decode token IDs back to text
    fn decode_tokens(&self, ids: &[u32]) -> String {
        let mut pieces = Vec::new();

        for &id in ids {
            // Look up token in vocabulary
            if let Some(token) = self.id_to_token.get(&id) {
                pieces.push(token.clone());
            } else {
                // Skip unknown tokens or use replacement
                if id >= 2 {
                    // Try to decode as byte for backward compatibility
                    let byte = (id - 2) as u8;
                    if byte.is_ascii() && !byte.is_ascii_control() {
                        if let Ok(s) = std::str::from_utf8(&[byte]) {
                            pieces.push(s.to_string());
                        }
                    }
                }
            }
        }

        pieces.concat()
    }
}

impl Default for GgufTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Tokenizer for GgufTokenizer {
    async fn tokenize(&self, text: &str, add_special: bool) -> Result<TokenizationResult> {
        let ids = self.tokenize_bytes(text);

        let tokens: Vec<String> = ids
            .iter()
            .map(|&id| {
                self.id_to_token
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("<token_{}>", id))
            })
            .collect();

        Ok(TokenizationResult::new(ids, tokens))
    }

    async fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.decode_tokens(ids))
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
    async fn test_gguf_tokenizer() {
        let tokenizer = GgufTokenizer::new();

        let result = tokenizer.tokenize("hello", false).await.unwrap();
        assert!(!result.ids.is_empty());

        let decoded = tokenizer.decode(&result.ids).await.unwrap();
        assert!(!decoded.is_empty());
    }
}
