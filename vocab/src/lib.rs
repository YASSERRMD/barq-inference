#![allow(
    clippy::all,
    unexpected_cfgs,
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code
)]
#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code,
    clippy::needless_update,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::let_and_return,
    clippy::manual_range_contains
)]

//! Vocabulary and tokenization implementations
//!
//! This module provides support for multiple tokenization methods used in
//! modern LLMs: SentencePiece, BPE, WordPiece, Unigram, and more.

pub mod bpe;
pub mod gguf_tokenizer;
pub mod prelude;
pub mod spm;
pub mod tokenizer;
pub mod unigram;
pub mod vocab;
pub mod wpm;

pub use bpe::BpeTokenizer;
pub use gguf_tokenizer::GgufTokenizer;
pub use spm::SentencePiece;
pub use tokenizer::{Tokenizer, TokenizerType};
pub use unigram::Unigram;
pub use vocab::{SpecialToken, Token, Vocab};
pub use wpm::WordPiece;
