//! GBNF grammar parser
//!
//! Parses GBNF (GGML Backus-Naur Form) grammar strings into Grammar structures.

use super::{Grammar, GrammarError, GrammarRule, Result};
use std::collections::HashMap;

/// GBNF Grammar Parser
pub struct GrammarParser;

impl GrammarParser {
    /// Parse a GBNF grammar string
    pub fn parse(input: &str) -> Result<Grammar> {
        let mut rules = HashMap::new();
        let mut root = String::new();

        for (line_num, line) in input.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse rule: name ::= definition
            if let Some(pos) = line.find("::=") {
                let name = line[..pos].trim().to_string();
                let definition = line[pos + 3..].trim().to_string();

                if root.is_empty() {
                    root = name.clone();
                }

                let rule = Self::parse_rule(&definition)?;
                rules.insert(name, rule);
            } else {
                return Err(GrammarError::ParseError(format!(
                    "Invalid syntax at line {}: {}",
                    line_num + 1,
                    line
                )));
            }
        }

        if root.is_empty() {
            return Err(GrammarError::ParseError("No rules found".to_string()));
        }

        Ok(Grammar { rules, root })
    }

    /// Parse a rule definition
    fn parse_rule(def: &str) -> Result<GrammarRule> {
        let def = def.trim();

        // Handle quotes (terminal strings)
        if def.starts_with('"') && def.ends_with('"') {
            let content = &def[1..def.len() - 1];
            return Ok(GrammarRule::Terminal(content.to_string()));
        }

        // Handle character classes [a-z]
        if def.starts_with('[') && def.ends_with(']') {
            let content = &def[1..def.len() - 1];
            return Ok(GrammarRule::Class(content.to_string()));
        }

        // Handle alternation (|)
        if def.contains('|') {
            let options: Vec<String> = def.split('|').map(|s| s.trim().to_string()).collect();
            return Ok(GrammarRule::Alt(options));
        }

        // Handle repetition *
        if def.ends_with('*') {
            let rule = &def[..def.len() - 1];
            return Ok(GrammarRule::Rep0(rule.trim().to_string()));
        }

        // Handle repetition +
        if def.ends_with('+') {
            let rule = &def[..def.len() - 1];
            return Ok(GrammarRule::Rep1(rule.trim().to_string()));
        }

        // Handle optional ?
        if def.ends_with('?') {
            let rule = &def[..def.len() - 1];
            return Ok(GrammarRule::Opt(rule.trim().to_string()));
        }

        // Handle sequences (space-separated)
        if def.contains(' ') {
            let parts: Vec<String> = def.split_whitespace().map(|s| s.to_string()).collect();
            return Ok(GrammarRule::Seq(parts));
        }

        // Otherwise it's a reference
        Ok(GrammarRule::Ref(def.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_rule() {
        let grammar = GrammarParser::parse("root ::= \"hello\"").unwrap();
        assert!(grammar.rules.contains_key("root"));
        assert_eq!(grammar.root, "root");
    }

    #[test]
    fn test_parse_alternation() {
        let grammar = GrammarParser::parse("root ::= \"hello\" | \"world\"").unwrap();
        assert!(grammar.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_character_class() {
        let grammar = GrammarParser::parse("root ::= [a-z]").unwrap();
        assert!(grammar.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_multiple_rules() {
        let input = r#"
            root ::= greeting " " name
            greeting ::= "hello" | "hi"
            name ::= [a-z]+
        "#;

        let grammar = GrammarParser::parse(input).unwrap();
        assert_eq!(grammar.rules.len(), 3);
        assert_eq!(grammar.root, "root");
    }
}
