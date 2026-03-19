//! Grammar-guided sampling
//!
//! Integrates GBNF grammar constraints with token sampling to ensure
//! generated output conforms to specified grammar rules.

use barq_core::error::{Error, Result as CoreResult};
use barq_core::grammar::{
    Grammar, GrammarCompiler, GrammarError, GrammarParser, GrammarRule, Result as GrammarResult,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use vocab::Vocab;

use crate::sampler::{Sampler, SamplerType, TokenData};

/// Grammar-guided sampler
///
/// Constrains token sampling to only valid tokens according to grammar.
#[derive(Debug, Clone)]
pub struct GrammarSampler {
    /// Grammar being enforced
    grammar: Arc<Grammar>,
    /// Compiled grammar masks
    masks: Arc<HashMap<String, Vec<bool>>>,
    /// Vocabulary size
    vocab_size: usize,
    /// Current rule stack
    rule_stack: Vec<String>,
    /// Current position in each rule
    positions: Vec<usize>,
}

impl GrammarSampler {
    /// Create a new grammar sampler using the placeholder compiler.
    pub fn new(grammar: Grammar, vocab_size: usize) -> GrammarResult<Self> {
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

    /// Create a new grammar sampler using the actual vocabulary tokens.
    ///
    /// This uses token text to approximate grammar-aware masks so structured
    /// modes such as JSON can constrain the first emitted token.
    pub fn new_with_vocab(grammar: Grammar, vocab: &Vocab) -> GrammarResult<Self> {
        let root = grammar.root.clone();
        let masks = Self::compile_masks_with_vocab(&grammar, vocab);

        Ok(Self {
            grammar: Arc::new(grammar),
            masks: Arc::new(masks),
            vocab_size: vocab.len(),
            rule_stack: vec![root],
            positions: vec![0],
        })
    }

    fn compile_masks_with_vocab(grammar: &Grammar, vocab: &Vocab) -> HashMap<String, Vec<bool>> {
        let mut masks = HashMap::new();

        for rule_name in grammar.rules.keys() {
            let mut mask = vec![false; vocab.len()];

            for token in vocab.iter() {
                if Self::token_matches_rule(grammar, rule_name, &token.text) {
                    if let Some(slot) = mask.get_mut(token.id as usize) {
                        *slot = true;
                    }
                }
            }

            // Avoid dead-end masks that would make generation impossible.
            if !mask.iter().any(|allowed| *allowed) {
                mask.fill(true);
            }

            masks.insert(rule_name.clone(), mask);
        }

        masks
    }

    fn token_matches_rule(grammar: &Grammar, rule_name: &str, token_text: &str) -> bool {
        let mut seen = HashSet::new();
        Self::rule_matches_token(grammar, rule_name, token_text, &mut seen, 0)
    }

    fn rule_matches_token(
        grammar: &Grammar,
        rule_name: &str,
        token_text: &str,
        seen: &mut HashSet<String>,
        depth: usize,
    ) -> bool {
        if depth > 16 {
            return true;
        }

        if !seen.insert(rule_name.to_string()) {
            return true;
        }

        let result = match grammar.rules.get(rule_name) {
            Some(GrammarRule::Terminal(terminal)) => {
                Self::token_matches_terminal(token_text, terminal)
            }
            Some(GrammarRule::Class(class)) => Self::token_matches_class(token_text, class),
            Some(GrammarRule::Seq(parts)) => parts
                .iter()
                .any(|part| Self::part_matches(grammar, part, token_text, seen, depth + 1)),
            Some(GrammarRule::Alt(parts)) => parts
                .iter()
                .any(|part| Self::part_matches(grammar, part, token_text, seen, depth + 1)),
            Some(GrammarRule::Opt(rule)) => {
                Self::part_matches(grammar, rule, token_text, seen, depth + 1)
            }
            Some(GrammarRule::Rep0(rule)) => {
                Self::part_matches(grammar, rule, token_text, seen, depth + 1)
            }
            Some(GrammarRule::Rep1(rule)) => {
                Self::part_matches(grammar, rule, token_text, seen, depth + 1)
            }
            Some(GrammarRule::Ref(name)) => {
                Self::rule_matches_token(grammar, name, token_text, seen, depth + 1)
            }
            None => true,
        };

        seen.remove(rule_name);
        result
    }

    fn part_matches(
        grammar: &Grammar,
        part: &str,
        token_text: &str,
        seen: &mut HashSet<String>,
        depth: usize,
    ) -> bool {
        let part = part.trim();
        if part.is_empty() {
            return true;
        }

        if let Some(terminal) = Self::strip_quotes(part) {
            return Self::token_matches_terminal(token_text, terminal);
        }

        if Self::looks_like_class(part) {
            return Self::token_matches_class(token_text, part.trim_matches(['[', ']']));
        }

        if grammar.rules.contains_key(part) {
            return Self::rule_matches_token(grammar, part, token_text, seen, depth + 1);
        }

        Self::token_matches_terminal(token_text, part)
    }

    fn strip_quotes(input: &str) -> Option<&str> {
        let input = input.trim();
        if input.len() >= 2 && input.starts_with('"') && input.ends_with('"') {
            Some(&input[1..input.len() - 1])
        } else {
            None
        }
    }

    fn normalize_text(input: &str) -> Vec<String> {
        let trimmed = input.trim();
        let stripped = trimmed
            .trim_start_matches(|c: char| c == '▁' || c == 'Ġ')
            .trim_start();

        if stripped == trimmed {
            vec![trimmed.to_string()]
        } else {
            vec![trimmed.to_string(), stripped.to_string()]
        }
    }

    fn looks_like_class(input: &str) -> bool {
        input.starts_with('[') && input.ends_with(']')
    }

    fn token_matches_terminal(token_text: &str, terminal: &str) -> bool {
        if terminal.is_empty() {
            return true;
        }

        let token_variants = Self::normalize_text(token_text);
        let terminal_variants = Self::normalize_text(terminal);

        token_variants.iter().any(|token| {
            terminal_variants.iter().any(|expected| {
                token == expected || token.starts_with(expected) || expected.starts_with(token)
            })
        })
    }

    fn token_matches_class(token_text: &str, class_spec: &str) -> bool {
        let token_text = token_text
            .trim_start_matches(|c: char| c == '▁' || c == 'Ġ')
            .trim_start();
        let Some(first) = token_text.chars().next() else {
            return false;
        };

        let class_spec = class_spec.trim();
        if class_spec.is_empty() {
            return false;
        }

        let (negated, spec) = if let Some(rest) = class_spec.strip_prefix('^') {
            (true, rest)
        } else {
            (false, class_spec)
        };

        let matches = Self::class_contains(spec, first);
        if negated {
            !matches
        } else {
            matches
        }
    }

    fn class_contains(spec: &str, ch: char) -> bool {
        let mut chars = spec.chars().peekable();

        while let Some(start) = chars.next() {
            if chars.peek() == Some(&'-') {
                chars.next();
                if let Some(end) = chars.next() {
                    if start <= ch && ch <= end {
                        return true;
                    }
                    continue;
                }
                if start == ch {
                    return true;
                }
                break;
            }

            if start == ch {
                return true;
            }
        }

        false
    }

    fn current_mask(&self) -> Vec<bool> {
        if let Some(current_rule) = self.rule_stack.last() {
            self.masks
                .get(current_rule)
                .cloned()
                .unwrap_or_else(|| vec![true; self.vocab_size])
        } else {
            vec![true; self.vocab_size]
        }
    }

    /// Get current allowed tokens
    pub fn get_allowed_tokens(&self) -> Vec<usize> {
        self.current_mask()
            .iter()
            .enumerate()
            .filter(|(_, &allowed)| allowed)
            .map(|(id, _)| id)
            .collect()
    }

    /// Apply grammar constraints to logits
    ///
    /// Sets logits of disallowed tokens to -inf.
    pub fn apply_constraints(&self, logits: &mut [f32]) -> GrammarResult<()> {
        let mask = self.current_mask();
        let limit = logits.len().min(mask.len());

        for i in 0..limit {
            if !mask[i] {
                logits[i] = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Update state after accepting a token.
    pub fn accept_token(&mut self, _token_id: usize, _token_str: &str) -> GrammarResult<()> {
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

    /// Check if grammar is complete.
    ///
    /// This is a conservative check for the current lightweight state tracker.
    pub fn is_complete(&self) -> bool {
        self.positions.last().copied().unwrap_or(0) > 0
    }
}

impl Sampler for GrammarSampler {
    fn sample(&mut self, logits: &mut [TokenData]) -> CoreResult<i32> {
        if logits.is_empty() {
            return Err(Error::tensor("Empty logits"));
        }

        let mask = self.current_mask();
        let mut best: Option<TokenData> = None;

        for (idx, token) in logits.iter_mut().enumerate() {
            let allowed = mask.get(idx).copied().unwrap_or(true);
            if !allowed {
                token.logit = f32::NEG_INFINITY;
                token.p = 0.0;
                continue;
            }

            if best
                .as_ref()
                .map(|current| token.logit > current.logit)
                .unwrap_or(true)
            {
                best = Some(*token);
            }
        }

        best.map(|token| token.id)
            .ok_or_else(|| Error::tensor("No allowed tokens"))
    }

    fn reset(&mut self) {
        GrammarSampler::reset(self);
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Custom
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
    pub fn from_gbnf(&self, gbnf: &str) -> GrammarResult<GrammarSampler> {
        let grammar = GrammarParser::parse(gbnf)?;
        GrammarSampler::new(grammar, self.vocab_size)
    }

    /// Build token-aware sampler from grammar string
    pub fn from_gbnf_with_vocab(&self, gbnf: &str, vocab: &Vocab) -> GrammarResult<GrammarSampler> {
        let grammar = GrammarParser::parse(gbnf)?;
        GrammarSampler::new_with_vocab(grammar, vocab)
    }

    /// Build sampler from pre-parsed grammar
    pub fn from_grammar(&self, grammar: Grammar) -> GrammarResult<GrammarSampler> {
        GrammarSampler::new(grammar, self.vocab_size)
    }

    /// Build token-aware sampler from pre-parsed grammar
    pub fn from_grammar_with_vocab(
        &self,
        grammar: Grammar,
        vocab: &Vocab,
    ) -> GrammarResult<GrammarSampler> {
        GrammarSampler::new_with_vocab(grammar, vocab)
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
        assert_eq!(allowed.len(), 100);
    }

    #[test]
    fn test_grammar_sampler_with_vocab() {
        use vocab::{Token, TokenAttr, TokenType, Vocab, VocabType};

        let mut vocab = Vocab::new(VocabType::SPM);
        vocab.add_token(Token {
            id: 0,
            text: "hello".to_string(),
            score: 0.0,
            token_type: TokenType::Normal,
            attrs: TokenAttr::default(),
        });
        vocab.add_token(Token {
            id: 1,
            text: "world".to_string(),
            score: 0.0,
            token_type: TokenType::Normal,
            attrs: TokenAttr::default(),
        });

        let sampler = GrammarSamplerBuilder::new(2)
            .from_gbnf_with_vocab("root ::= \"hello\"", &vocab)
            .unwrap();

        let allowed = sampler.get_allowed_tokens();
        assert_eq!(allowed, vec![0]);
    }
}
