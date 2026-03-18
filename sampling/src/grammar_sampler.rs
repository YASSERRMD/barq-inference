//! Grammar-guided sampling
//!
//! Integrates GBNF grammar constraints with token sampling to ensure
//! generated output conforms to specified grammar rules.

use barq_core::grammar::{Grammar, GrammarCompiler, GrammarError};
use barq_core::tensor::Tensor;
use std::sync::Arc;

/// Grammar-guided sampler
///
/// Constrains token sampling to only valid tokens according to grammar
pub struct GrammarSampler {
    /// Grammar being enforced
    grammar: Arc<Grammar>,
    /// Compiled grammar masks
    masks: Arc<std::collections::HashMap<String, Vec<bool>>>,
    /// Vocabulary size
    vocab_size: usize,
    /// Current rule stack
    rule_stack: Vec<String>,
    /// Current position in each rule
    positions: Vec<usize>,
}

impl GrammarSampler {
    /// Create a new grammar sampler
    pub fn new(grammar: Grammar, vocab_size: usize) -> Result<Self, GrammarError> {
        let root = grammar.root.clone();
        let compiler = GrammarCompiler;
        let masks = compiler.compile(&grammar, vocab_size)?;

        Ok(Self {
            grammar: Arc::new(grammar),
            masks: Arc::new(masks),
            vocab_size,
            rule_stack: vec![root],
            positions: vec![0],
        })
    }

    /// Get current allowed tokens
    pub fn get_allowed_tokens(&self) -> Vec<usize> {
        if let Some(current_rule) = self.rule_stack.last() {
            self.masks
                .get(current_rule)
                .map(|mask| {
                    mask.iter()
                        .enumerate()
                        .filter(|(_, &allowed)| allowed)
                        .map(|(id, _)| id)
                        .collect()
                })
                .unwrap_or_else(|| {
                    // If no mask, allow all tokens
                    (0..self.vocab_size).collect()
                })
        } else {
            (0..self.vocab_size).collect()
        }
    }

    /// Apply grammar constraints to logits
    ///
    /// Sets logits of disallowed tokens to -inf
    pub fn apply_constraints(&self, logits: &mut [f32]) -> Result<(), GrammarError> {
        let allowed = self.get_allowed_tokens();
        let vocab_size = self.vocab_size;

        // Set disallowed tokens to -inf
        for i in 0..vocab_size {
            if !allowed.contains(&i) {
                logits[i] = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Update state after accepting a token
    pub fn accept_token(&mut self, token_id: usize, token_str: &str) -> Result<(), GrammarError> {
        // Update state based on token
        // In a full implementation, this would:
        // 1. Determine which rule(s) the token satisfies
        // 2. Advance position in current rule
        // 3. Push/pop rules from stack as needed

        // For now, just advance position
        if let Some(pos) = self.positions.last_mut() {
            *pos += 1;
        }

        Ok(())
    }

    /// Reset sampler state
    pub fn reset(&mut self) {
        self.rule_stack = vec![self.grammar.root.clone()];
        self.positions = vec![0];
    }

    /// Check if grammar is complete
    pub fn is_complete(&self) -> bool {
        // Grammar is complete if rule stack is empty
        // or if we're at end of all rules
        self.rule_stack.is_empty()
    }
}

/// Grammar sampler builder
pub struct GrammarSamplerBuilder {
    vocab_size: usize,
}

impl GrammarSamplerBuilder {
    /// Create a new builder
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Build sampler from grammar string
    pub fn from_gbnf(&self, gbnf: &str) -> Result<GrammarSampler, GrammarError> {
        use barq_core::grammar::GrammarParser;

        let grammar = GrammarParser::parse(gbnf)?;
        GrammarSampler::new(grammar, self.vocab_size)
    }

    /// Build sampler from pre-parsed grammar
    pub fn from_grammar(&self, grammar: Grammar) -> Result<GrammarSampler, GrammarError> {
        GrammarSampler::new(grammar, self.vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_sampler_creation() {
        use barq_core::grammar::{Grammar, GrammarRule};
        use std::collections::HashMap;

        let mut rules = HashMap::new();
        rules.insert(
            "root".to_string(),
            GrammarRule::Terminal("hello".to_string()),
        );

        let grammar = Grammar {
            rules,
            root: "root".to_string(),
        };

        let sampler = GrammarSampler::new(grammar, 1000).unwrap();
        assert_eq!(sampler.vocab_size, 1000);
    }

    #[test]
    fn test_grammar_sampler_from_gbnf() {
        let builder = GrammarSamplerBuilder::new(1000);
        let sampler = builder.from_gbnf("root ::= \"hello\"").unwrap();
        assert_eq!(sampler.vocab_size, 1000);
    }

    #[test]
    fn test_get_allowed_tokens() {
        use barq_core::grammar::{Grammar, GrammarRule};
        use std::collections::HashMap;

        let mut rules = HashMap::new();
        rules.insert(
            "root".to_string(),
            GrammarRule::Terminal("hello".to_string()),
        );

        let grammar = Grammar {
            rules,
            root: "root".to_string(),
        };

        let sampler = GrammarSampler::new(grammar, 100).unwrap();
        let allowed = sampler.get_allowed_tokens();
        // Should allow all tokens for now (masks not fully implemented)
        assert_eq!(allowed.len(), 100);
    }
}
