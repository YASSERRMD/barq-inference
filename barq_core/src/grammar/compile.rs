//! Grammar compilation
//!
//! Compiles grammar AST into token constraints for sampling.

use super::{Grammar, GrammarError, Result};

/// Grammar compiler
///
/// Compiles a parsed grammar into token constraints that can be used
/// during sampling to restrict output to valid grammar matches.
pub struct GrammarCompiler;

impl GrammarCompiler {
    /// Compile grammar into token masks
    ///
    /// Returns a mapping from grammar positions to valid token IDs
    pub fn compile(
        &self,
        grammar: &Grammar,
        vocab_size: usize,
    ) -> Result<std::collections::HashMap<String, Vec<bool>>> {
        let mut masks = std::collections::HashMap::new();

        for (rule_name, _) in &grammar.rules {
            // For now, create all-allowed masks
            // In a full implementation, this would analyze the grammar
            // and compute valid tokens for each rule position
            let mask = vec![true; vocab_size];
            masks.insert(rule_name.clone(), mask);
        }

        Ok(masks)
    }

    /// Get valid tokens for a rule at current position
    pub fn get_valid_tokens(
        &self,
        rule_name: &str,
        _position: usize,
        masks: &std::collections::HashMap<String, Vec<bool>>,
    ) -> Vec<usize> {
        masks
            .get(rule_name)
            .map(|mask| {
                mask.iter()
                    .enumerate()
                    .filter(|(_, &allowed)| allowed)
                    .map(|(id, _)| id)
                    .collect()
            })
            .unwrap_or_else(Vec::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_grammar() {
        let mut grammar = Grammar {
            rules: std::collections::HashMap::new(),
            root: "root".to_string(),
        };

        use crate::grammar::GrammarRule;
        grammar.rules.insert(
            "root".to_string(),
            GrammarRule::Terminal("hello".to_string()),
        );

        let compiler = GrammarCompiler;
        let masks = compiler.compile(&grammar, 1000).unwrap();

        assert!(masks.contains_key("root"));
        assert_eq!(masks["root"].len(), 1000);
    }

    #[test]
    fn test_get_valid_tokens() {
        let mut masks = std::collections::HashMap::new();
        let mut mask = vec![false; 10];
        mask[0] = true;
        mask[5] = true;
        masks.insert("root".to_string(), mask);

        let compiler = GrammarCompiler;
        let valid = compiler.get_valid_tokens("root", 0, &masks);

        assert_eq!(valid, vec![0, 5]);
    }
}
