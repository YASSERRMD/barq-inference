//! Vocabulary and token types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Token ID type
pub type TokenId = u32;

/// A single token in the vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// Token ID
    pub id: TokenId,
    /// Token text/score
    pub text: String,
    /// Token score (for ranking)
    pub score: f32,
    /// Token type
    pub token_type: TokenType,
    /// Token attributes
    pub attrs: TokenAttr,
}

/// Token type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenType {
    /// Normal token
    Normal,
    /// Unknown token
    Unknown,
    /// Control token (special tokens like BOS, EOS)
    Control,
    /// User-defined token
    UserDefined,
    /// Unused token
    Unused,
    /// Byte token
    Byte,
}

/// Token attributes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenAttr {
    /// Normalized token
    pub normalized: bool,
    /// Leading space stripped
    pub lstrip: bool,
    /// Trailing space stripped
    pub rstrip: bool,
    /// Single word
    pub single_word: bool,
}

impl Default for TokenAttr {
    fn default() -> Self {
        Self {
            normalized: false,
            lstrip: false,
            rstrip: false,
            single_word: false,
        }
    }
}

/// Special tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialToken {
    /// Beginning of sentence
    pub bos: Option<TokenId>,
    /// End of sentence
    pub eos: Option<TokenId>,
    /// End of turn
    pub eot: Option<TokenId>,
    /// Sentence separator
    pub sep: Option<TokenId>,
    /// Newline
    pub nl: Option<TokenId>,
    /// Padding
    pub pad: Option<TokenId>,
    /// Mask
    pub mask: Option<TokenId>,
    /// Prefix for infill
    pub fim_pre: Option<TokenId>,
    /// Suffix for infill
    pub fim_suf: Option<TokenId>,
    /// Middle for infill
    pub fim_mid: Option<TokenId>,
    /// Pad for infill
    pub fim_pad: Option<TokenId>,
}

impl Default for SpecialToken {
    fn default() -> Self {
        Self {
            bos: None,
            eos: None,
            eot: None,
            sep: None,
            nl: None,
            pad: None,
            mask: None,
            fim_pre: None,
            fim_suf: None,
            fim_mid: None,
            fim_pad: None,
        }
    }
}

/// Vocabulary
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Token to ID mapping
    token_to_id: HashMap<String, TokenId>,
    /// ID to token mapping
    id_to_token: HashMap<TokenId, Token>,
    /// Special tokens
    special: SpecialToken,
    /// Add BOS token automatically
    pub add_bos: bool,
    /// Add EOS token automatically
    pub add_eos: bool,
    /// Vocabulary type
    pub vocab_type: VocabType,
}

/// Vocabulary type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocabType {
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

impl Vocab {
    /// Create a new vocabulary
    pub fn new(vocab_type: VocabType) -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special: SpecialToken::default(),
            add_bos: false,
            add_eos: false,
            vocab_type,
        }
    }

    /// Add a token to the vocabulary
    pub fn add_token(&mut self, token: Token) {
        self.token_to_id.insert(token.text.clone(), token.id);
        self.id_to_token.insert(token.id, token);
    }

    /// Get token ID from text
    pub fn get_id(&self, text: &str) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }

    /// Get token from ID
    pub fn get_token(&self, id: TokenId) -> Option<&Token> {
        self.id_to_token.get(&id)
    }

    /// Returns the vocabulary size
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Returns true if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Set special tokens
    pub fn set_special_tokens(&mut self, special: SpecialToken) {
        self.special = special;
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialToken {
        &self.special
    }

    /// Find all tokens starting with a prefix
    pub fn find_tokens_with_prefix(&self, prefix: &str) -> Vec<&Token> {
        self.token_to_id
            .keys()
            .filter(|k| k.starts_with(prefix))
            .filter_map(|k| self.get_token(self.token_to_id[k]))
            .collect()
    }
}

/// Tokenization result
#[derive(Debug, Clone)]
pub struct TokenizationResult {
    /// Token IDs
    pub ids: Vec<TokenId>,
    /// Token texts
    pub tokens: Vec<String>,
    /// Number of tokens
    pub len: usize,
}

impl TokenizationResult {
    /// Create a new tokenization result
    pub fn new(ids: Vec<TokenId>, tokens: Vec<String>) -> Self {
        let len = ids.len();
        Self { ids, tokens, len }
    }

    /// Returns the number of tokens
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab() {
        let mut vocab = Vocab::new(VocabType::SPM);

        vocab.add_token(Token {
            id: 0,
            text: "<unk>".to_string(),
            score: 0.0,
            token_type: TokenType::Unknown,
            attrs: TokenAttr::default(),
        });

        vocab.add_token(Token {
            id: 1,
            text: "hello".to_string(),
            score: -1.0,
            token_type: TokenType::Normal,
            attrs: TokenAttr::default(),
        });

        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.get_id("hello"), Some(1));
        assert_eq!(vocab.get_token(1).unwrap().text, "hello");
    }
}
