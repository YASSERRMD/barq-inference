//! GGUF tokenizer implementation

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use rust_tokenizers::tokenizer::{
    Gpt2Tokenizer as RtGpt2Tokenizer, Tokenizer as RtTokenizer, TruncationStrategy,
};

use crate::tokenizer::{Tokenizer, TokenizerType};
use crate::vocab::{
    SpecialToken, Token, TokenAttr, TokenType, TokenizationResult, Vocab, VocabType,
};

enum GgufTokenizerBackend {
    SentencePiece,
    Gpt2 {
        tokenizer: RtGpt2Tokenizer,
        add_bos_token: bool,
    },
}

#[derive(Debug, Clone, Copy)]
struct SpecialTokenIds {
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
}

/// Simple GGUF tokenizer
pub struct GgufTokenizer {
    vocab: Arc<Vocab>,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    backend: GgufTokenizerBackend,
}

impl GgufTokenizer {
    fn add_vocab_token(vocab: &mut Vocab, id: u32, text: &str, token_type: TokenType) {
        vocab.add_token(Token {
            id,
            text: text.to_string(),
            score: 0.0,
            token_type,
            attrs: TokenAttr::default(),
        });
    }

    fn parse_json_vec<T>(metadata: &HashMap<String, String>, key: &str) -> Option<Vec<T>>
    where
        T: serde::de::DeserializeOwned,
    {
        metadata
            .get(key)
            .and_then(|value| serde_json::from_str::<Vec<T>>(value).ok())
    }

    fn parse_bool(metadata: &HashMap<String, String>, key: &str, default: bool) -> bool {
        metadata
            .get(key)
            .and_then(|value| value.parse::<bool>().ok())
            .unwrap_or(default)
    }

    fn parse_u32(metadata: &HashMap<String, String>, key: &str) -> Option<u32> {
        metadata
            .get(key)
            .and_then(|value| value.parse::<u32>().ok())
    }

    /// Load tokenizer from GGUF model metadata
    pub fn from_gguf(metadata: &HashMap<String, String>) -> Self {
        if matches!(
            metadata
                .get("tokenizer.ggml.model")
                .map(|value| value.as_str()),
            Some("gpt2")
        ) {
            if let Some(tokenizer) = Self::from_gpt2_gguf(metadata) {
                return tokenizer;
            }
        }

        Self::from_sentencepiece_gguf(metadata)
    }

    fn build_vocab_from_tokens(
        tokens: &[String],
        token_types: Option<&[i32]>,
        vocab_type: VocabType,
        special_ids: SpecialTokenIds,
        add_bos_token: bool,
    ) -> (Arc<Vocab>, HashMap<String, u32>, HashMap<u32, String>) {
        let mut vocab = Vocab::new(vocab_type);
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in tokens.iter().enumerate() {
            let id_u32 = id as u32;
            token_to_id.insert(token.clone(), id_u32);
            id_to_token.insert(id_u32, token.clone());

            let token_type = match token_types.and_then(|types| types.get(id).copied()) {
                Some(3) => TokenType::Control,
                Some(4) => TokenType::UserDefined,
                Some(5) => TokenType::Byte,
                Some(_) => TokenType::Normal,
                None if token.starts_with('<') && token.ends_with('>') => TokenType::Control,
                None => TokenType::Normal,
            };

            Self::add_vocab_token(&mut vocab, id_u32, token, token_type);
        }

        vocab.add_bos = add_bos_token;
        vocab.add_eos = special_ids.eos.is_some();
        vocab.set_special_tokens(SpecialToken {
            bos: special_ids.bos,
            eos: special_ids.eos,
            pad: special_ids.pad,
            ..Default::default()
        });

        (Arc::new(vocab), token_to_id, id_to_token)
    }

    fn build_special_token_map(
        tokens: &[String],
        token_types: Option<&[i32]>,
        special_ids: SpecialTokenIds,
    ) -> serde_json::Value {
        let mut additional_special_tokens = HashSet::new();
        for (idx, token) in tokens.iter().enumerate() {
            let token_type = token_types.and_then(|types| types.get(idx)).copied();
            if matches!(token_type, Some(3) | Some(4)) {
                additional_special_tokens.insert(token.clone());
            }
        }

        let unk_token = special_ids
            .bos
            .and_then(|id| tokens.get(id as usize).cloned())
            .or_else(|| {
                special_ids
                    .eos
                    .and_then(|id| tokens.get(id as usize).cloned())
            })
            .or_else(|| tokens.get(0).cloned())
            .unwrap_or_else(|| "<|endoftext|>".to_string());

        let bos_token = special_ids
            .bos
            .and_then(|id| tokens.get(id as usize).cloned());
        let eos_token = special_ids
            .eos
            .and_then(|id| tokens.get(id as usize).cloned());
        let pad_token = special_ids
            .pad
            .and_then(|id| tokens.get(id as usize).cloned());

        let additional_special_tokens = if additional_special_tokens.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::json!(additional_special_tokens)
        };

        serde_json::json!({
            "unk_token": unk_token,
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "mask_token": serde_json::Value::Null,
            "sep_token": serde_json::Value::Null,
            "cls_token": serde_json::Value::Null,
            "additional_special_tokens": additional_special_tokens,
        })
    }

    fn unique_temp_path(prefix: &str, suffix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "barq-{}-{}-{}.{}",
            prefix,
            std::process::id(),
            nanos,
            suffix.trim_start_matches('.')
        ))
    }

    fn build_gpt2_tokenizer(
        tokens: &[String],
        token_types: Option<&[i32]>,
        merges: &[String],
        special_ids: SpecialTokenIds,
    ) -> Result<RtGpt2Tokenizer> {
        let vocab_map: HashMap<String, i64> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as i64))
            .collect();
        let special_map = Self::build_special_token_map(tokens, token_types, special_ids);

        let vocab_path = Self::unique_temp_path("gpt2-vocab", "json");
        let merges_path = Self::unique_temp_path("gpt2-merges", "txt");
        let special_path = Self::unique_temp_path("gpt2-special", "json");

        fs::write(&vocab_path, serde_json::to_vec(&vocab_map)?)?;

        let mut merges_text = String::from("#version: 0.1\n");
        for merge in merges {
            merges_text.push_str(merge);
            merges_text.push('\n');
        }
        fs::write(&merges_path, merges_text)?;
        fs::write(&special_path, serde_json::to_vec(&special_map)?)?;

        let tokenizer = RtGpt2Tokenizer::from_file_with_special_token_mapping(
            &vocab_path,
            &merges_path,
            false,
            &special_path,
        )?;

        let _ = fs::remove_file(&vocab_path);
        let _ = fs::remove_file(&merges_path);
        let _ = fs::remove_file(&special_path);

        Ok(tokenizer)
    }

    fn from_sentencepiece_gguf(metadata: &HashMap<String, String>) -> Self {
        let mut vocab = Vocab::new(VocabType::SPM);
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
                    Self::add_vocab_token(
                        &mut vocab,
                        id_u32,
                        token,
                        if token.starts_with('<') && token.ends_with('>') {
                            TokenType::Control
                        } else {
                            TokenType::Normal
                        },
                    );
                }
                vocab.set_special_tokens(SpecialToken {
                    bos: token_to_id
                        .get("<s>")
                        .or_else(|| token_to_id.get("<bos>"))
                        .copied(),
                    eos: token_to_id
                        .get("</s>")
                        .or_else(|| token_to_id.get("<eos>"))
                        .copied(),
                    ..Default::default()
                });
                eprintln!("Loaded {} tokens into vocabulary", id_to_token.len());
                return Self {
                    vocab: Arc::new(vocab),
                    token_to_id,
                    id_to_token,
                    backend: GgufTokenizerBackend::SentencePiece,
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
                        Self::add_vocab_token(
                            &mut vocab,
                            id,
                            value,
                            if value.starts_with('<') && value.ends_with('>') {
                                TokenType::Control
                            } else {
                                TokenType::Normal
                            },
                        );
                    }
                }
            }
        }

        if id_to_token.is_empty() {
            eprintln!("Warning: No vocabulary loaded from GGUF metadata!");
            eprintln!("Available keys: {:?}", metadata.keys().collect::<Vec<_>>());
        }

        vocab.set_special_tokens(SpecialToken {
            bos: token_to_id
                .get("<s>")
                .or_else(|| token_to_id.get("<bos>"))
                .copied(),
            eos: token_to_id
                .get("</s>")
                .or_else(|| token_to_id.get("<eos>"))
                .copied(),
            ..Default::default()
        });

        Self {
            vocab: Arc::new(vocab),
            token_to_id,
            id_to_token,
            backend: GgufTokenizerBackend::SentencePiece,
        }
    }

    fn from_gpt2_gguf(metadata: &HashMap<String, String>) -> Option<Self> {
        let tokens = Self::parse_json_vec::<String>(metadata, "tokenizer.ggml.tokens")?;
        let merges = Self::parse_json_vec::<String>(metadata, "tokenizer.ggml.merges")?;
        let token_types = Self::parse_json_vec::<i32>(metadata, "tokenizer.ggml.token_type");
        let add_bos_token = Self::parse_bool(metadata, "tokenizer.ggml.add_bos_token", false);
        let special_ids = SpecialTokenIds {
            bos: Self::parse_u32(metadata, "tokenizer.ggml.bos_token_id"),
            eos: Self::parse_u32(metadata, "tokenizer.ggml.eos_token_id"),
            pad: Self::parse_u32(metadata, "tokenizer.ggml.padding_token_id"),
        };

        let (vocab, token_to_id, id_to_token) = Self::build_vocab_from_tokens(
            &tokens,
            token_types.as_deref(),
            VocabType::BPE,
            special_ids,
            add_bos_token,
        );

        let tokenizer =
            match Self::build_gpt2_tokenizer(&tokens, token_types.as_deref(), &merges, special_ids)
            {
                Ok(tokenizer) => tokenizer,
                Err(e) => {
                    eprintln!("Failed to build GPT-2 tokenizer from GGUF metadata: {e}");
                    return None;
                }
            };

        Some(Self {
            vocab,
            token_to_id,
            id_to_token,
            backend: GgufTokenizerBackend::Gpt2 {
                tokenizer,
                add_bos_token,
            },
        })
    }

    /// Create a new GGUF tokenizer
    pub fn new() -> Self {
        let mut vocab = Vocab::new(VocabType::SPM);
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Add basic tokens (placeholder)
        token_to_id.insert("<s>".to_string(), 0);
        token_to_id.insert("</s>".to_string(), 1);
        id_to_token.insert(0, "<s>".to_string());
        id_to_token.insert(1, "</s>".to_string());

        vocab.set_special_tokens(SpecialToken {
            bos: Some(0),
            eos: Some(1),
            ..Default::default()
        });
        Self::add_vocab_token(&mut vocab, 0, "<s>", TokenType::Control);
        Self::add_vocab_token(&mut vocab, 1, "</s>", TokenType::Control);

        Self {
            vocab: Arc::new(vocab),
            token_to_id,
            id_to_token,
            backend: GgufTokenizerBackend::SentencePiece,
        }
    }

    /// Simple byte-level tokenization
    fn tokenize_bytes(&self, text: &str) -> Vec<u32> {
        // Fallback or byte tokenization
        let mut tokens = Vec::new();
        for byte in text.bytes() {
            let token_id = 2 + (byte as u32);
            tokens.push(token_id);
        }
        tokens
    }

    /// Greedy longest-match tokenization for SPM
    fn tokenize_greedy(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut tokens = Vec::new();

        if add_special {
            // Usually 1 is <s> for LLaMA
            if let Some(&id) = self.token_to_id.get("<s>") {
                tokens.push(id);
            } else {
                tokens.push(1);
            }
        }

        // Replace spaces with standard SPM space block U+2581
        let spm_text = text.replace(' ', "\u{2581}");
        let mut check_text = String::new();
        if !text.starts_with(' ') {
            check_text.push_str("\u{2581}");
        }
        check_text.push_str(&spm_text);

        let mut i = 0;
        let chars: Vec<char> = check_text.chars().collect();

        while i < chars.len() {
            let mut match_len = 0;
            let mut match_id = 0;

            // Try all lengths in reverse to find longest match
            for len in (1..=chars.len() - i).rev() {
                let substr: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.token_to_id.get(&substr) {
                    match_len = len;
                    match_id = id;
                    break;
                }
            }

            if match_len > 0 {
                tokens.push(match_id);
                i += match_len;
            } else {
                // Unknown character fallback
                if let Some(&unk) = self.token_to_id.get("<unk>") {
                    tokens.push(unk);
                } else {
                    tokens.push(0);
                }
                i += 1;
            }
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
        match &self.backend {
            GgufTokenizerBackend::Gpt2 {
                tokenizer,
                add_bos_token,
            } => {
                let tokenized = tokenizer.encode(
                    text,
                    None,
                    usize::MAX,
                    &TruncationStrategy::DoNotTruncate,
                    0,
                );
                let mut ids = tokenized
                    .token_ids
                    .into_iter()
                    .map(|id| id as u32)
                    .collect::<Vec<_>>();

                if add_special && *add_bos_token {
                    if let Some(bos) = self.vocab.special_tokens().bos {
                        ids.insert(0, bos);
                    }
                }

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
            GgufTokenizerBackend::SentencePiece => {
                // If we only have dummy token IDs starting from 0, use byte tokenization.
                // Otherwise use our greedy SPM matching.
                let ids = if self.token_to_id.len() > 100 {
                    self.tokenize_greedy(text, add_special)
                } else {
                    self.tokenize_bytes(text)
                };

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
        }
    }

    async fn decode(&self, ids: &[u32]) -> Result<String> {
        match &self.backend {
            GgufTokenizerBackend::Gpt2 { tokenizer, .. } => {
                let ids_i64: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
                Ok(tokenizer.decode(&ids_i64, true, false))
            }
            GgufTokenizerBackend::SentencePiece => Ok(self.decode_tokens(ids)),
        }
    }

    fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    fn tokenizer_type(&self) -> TokenizerType {
        match self.backend {
            GgufTokenizerBackend::Gpt2 { .. } => TokenizerType::BPE,
            GgufTokenizerBackend::SentencePiece => TokenizerType::SPM,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    #[tokio::test]
    async fn test_gguf_tokenizer() {
        let tokenizer = GgufTokenizer::new();

        let result = tokenizer.tokenize("hello", false).await.unwrap();
        assert!(!result.ids.is_empty());

        let decoded = tokenizer.decode(&result.ids).await.unwrap();
        assert!(decoded.len() <= result.ids.len());
    }

    #[tokio::test]
    async fn test_gguf_tokenizer_vocab_is_populated() {
        let tokenizer = GgufTokenizer::new();

        assert!(tokenizer.vocab().len() >= 2);
        assert_eq!(tokenizer.vocab().special_tokens().bos, Some(0));
        assert_eq!(tokenizer.vocab().special_tokens().eos, Some(1));
    }

    #[tokio::test]
    async fn test_gguf_tokenizer_gpt2_backend_uses_special_tokens() {
        let metadata = HashMap::from([
            ("tokenizer.ggml.model".to_string(), "gpt2".to_string()),
            (
                "tokenizer.ggml.add_bos_token".to_string(),
                "false".to_string(),
            ),
            ("tokenizer.ggml.bos_token_id".to_string(), "0".to_string()),
            ("tokenizer.ggml.eos_token_id".to_string(), "2".to_string()),
            (
                "tokenizer.ggml.padding_token_id".to_string(),
                "0".to_string(),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                serde_json::to_string(&vec![
                    "<|endoftext|>".to_string(),
                    "<|im_start|>".to_string(),
                    "<|im_end|>".to_string(),
                    "h".to_string(),
                ])
                .unwrap(),
            ),
            (
                "tokenizer.ggml.merges".to_string(),
                serde_json::to_string(&Vec::<String>::new()).unwrap(),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                serde_json::to_string(&vec![3_i32, 3, 3, 1]).unwrap(),
            ),
        ]);

        let tokenizer = GgufTokenizer::from_gguf(&metadata);

        let tokenized = tokenizer
            .tokenize("<|im_start|>h<|im_end|>", true)
            .await
            .unwrap();

        assert_eq!(tokenized.ids, vec![1, 3, 2]);

        let decoded = tokenizer.decode(&tokenized.ids).await.unwrap();
        assert_eq!(decoded, "h");
        assert_eq!(tokenizer.tokenizer_type(), TokenizerType::BPE);
    }
}
