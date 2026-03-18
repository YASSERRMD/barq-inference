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
    pub fn from_gguf(metadata: &HashMap<String, String>) -> Self {
        let vocab = Arc::new(Vocab::new(VocabType::SPM));
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Try to load tokenizer tokens array (stored as JSON)
        if let Some(tokens_json) = metadata.get("tokenizer.ggml.tokens") {
            if let Ok(tokens_vec) = serde_json::from_str::<Vec<String>>(tokens_json) {
                eprintln!(
                    "Loading vocabulary with {} tokens from GGUF",
                    tokens_vec.len()
                );
                // Build the token mappings
                for (id, token) in tokens_vec.iter().enumerate() {
                    let id_u32 = id as u32;
                    id_to_token.insert(id_u32, token.clone());
                    token_to_id.insert(token.clone(), id_u32);
                }
                eprintln!("Loaded {} tokens into vocabulary", id_to_token.len());
                return Self {
                    vocab,
                    token_to_id,
                    id_to_token,
                };
            } else {
                eprintln!("Failed to parse tokenizer.ggml.tokens JSON");
            }
        }

        // Fallback: Try old format (individual token entries)
        for (key, value) in metadata {
            if key.starts_with("tokenizer.ggml.token.") {
                if let Some(id_str) = key.strip_prefix("tokenizer.ggml.token.") {
                    if let Ok(id) = id_str.parse::<u32>() {
                        id_to_token.insert(id, value.clone());
                        token_to_id.insert(value.clone(), id);
                    }
                }
            }
        }

        if id_to_token.is_empty() {
            eprintln!("Warning: No vocabulary loaded from GGUF metadata!");
            eprintln!("Available keys: {:?}", metadata.keys().collect::<Vec<_>>());
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
        let mut bytes = Vec::new();

        for &id in ids {
            // Look up token in vocabulary
            if let Some(token) = self.id_to_token.get(&id) {
                // Handle byte-level tokens like "<0x00>"
                if token.starts_with("<0x") && token.ends_with('>') {
                    // Parse hex byte value
                    if let Ok(byte_val) = u8::from_str_radix(&token[3..token.len() - 1], 16) {
                        bytes.push(byte_val);
                    }
                } else if token == "<unk>" || token == "<s>" || token == "</s>" {
                    // Skip special tokens
                    continue;
                } else {
                    // Regular text token - add as bytes
                    bytes.extend_from_slice(token.as_bytes());
                }
            }
        }

        // Convert bytes to UTF-8 string, replacing invalid sequences
        String::from_utf8_lossy(&bytes).to_string()
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
