//! Grammar system for structured output generation
//!
//! This module provides GBNF (GBNF = GBNF Backus-Naur Form) grammar parsing
//! and grammar-guided sampling for constrained text generation.
//!
//! # Features
//!
//! - Parse GBNF grammar from strings
//! - Build grammar AST
//! - Compile grammar to token constraints
//! - Apply grammar constraints during sampling
//! - JSON schema to GBNF conversion

pub mod ast;
pub mod compile;
pub mod parser;

pub use ast::*;
pub use compile::GrammarCompiler;
pub use parser::GrammarParser;

/// Error types for grammar operations
pub type Result<T> = std::result::Result<T, GrammarError>;

#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Compilation error: {0}")]
    CompilationError(String),

    #[error("Invalid grammar rule: {0}")]
    InvalidRule(String),

    #[error("Invalid JSON schema: {0}")]
    InvalidJsonSchema(String),

    #[error("Invalid JSON output: {0}")]
    InvalidJsonOutput(String),
}

/// GBNF Grammar
#[derive(Debug, Clone)]
pub struct Grammar {
    /// Grammar rules indexed by name
    pub rules: std::collections::HashMap<String, GrammarRule>,
    /// Root rule name
    pub root: String,
}

/// Grammar rule
#[derive(Debug, Clone)]
pub enum GrammarRule {
    /// Sequence of rules (concatenation)
    Seq(Vec<String>),

    /// Choice of rules (alternation)
    Alt(Vec<String>),

    /// Optional rule
    Opt(String),

    /// Repeat rule zero or more times
    Rep0(String),

    /// Repeat rule one or more times
    Rep1(String),

    /// Terminal string literal
    Terminal(String),

    /// Character class [a-z]
    Class(String),

    /// Reference to another rule
    Ref(String),
}

/// Grammar position for tracking parse state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrammarPos {
    pub rule_id: usize,
    pub position: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_creation() {
        let mut grammar = Grammar {
            rules: std::collections::HashMap::new(),
            root: "root".to_string(),
        };

        grammar.rules.insert(
            "root".to_string(),
            GrammarRule::Terminal("hello".to_string()),
        );

        assert!(grammar.rules.contains_key("root"));
    }
}
