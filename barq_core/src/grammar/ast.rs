//! Grammar AST types
//!
//! Abstract Syntax Tree representations for grammar rules.

use super::{Grammar, GrammarError, GrammarRule, Result};

/// Grammar AST node
#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    /// Terminal string literal
    Terminal(String),

    /// Character class [a-z]
    Class(String),

    /// Sequence of nodes (concatenation)
    Sequence(Vec<AstNode>),

    /// Choice of nodes (alternation)
    Choice(Vec<AstNode>),

    /// Optional node
    Optional(Box<AstNode>),

    /// Zero or more repetitions
    Repeat0(Box<AstNode>),

    /// One or more repetitions
    Repeat1(Box<AstNode>),

    /// Reference to another rule
    Ref(String),
}

impl AstNode {
    /// Check if this node can match empty string
    pub fn is_nullable(&self) -> bool {
        matches!(self, AstNode::Optional(_) | AstNode::Repeat0(_))
    }

    /// Get the minimum length this node can match
    pub fn min_len(&self) -> usize {
        match self {
            AstNode::Terminal(s) => s.len(),
            AstNode::Class(_) => 1,
            AstNode::Sequence(nodes) => nodes.iter().map(|n| n.min_len()).sum(),
            AstNode::Choice(nodes) => nodes.iter().map(|n| n.min_len()).min().unwrap_or(0),
            AstNode::Optional(_) | AstNode::Repeat0(_) => 0,
            AstNode::Repeat1(node) => node.min_len(),
            AstNode::Ref(_) => 0, // Can't determine without resolving
        }
    }

    /// Check if this node is a terminal
    pub fn is_terminal(&self) -> bool {
        matches!(self, AstNode::Terminal(_))
    }

    /// Check if this node is a reference
    pub fn is_ref(&self) -> bool {
        matches!(self, AstNode::Ref(_))
    }

    /// Convert GrammarRule to AstNode
    pub fn from_rule(rule: &GrammarRule) -> Result<Self> {
        match rule {
            GrammarRule::Terminal(s) => Ok(AstNode::Terminal(s.clone())),
            GrammarRule::Class(s) => Ok(AstNode::Class(s.clone())),
            GrammarRule::Seq(items) => {
                let mut nodes = Vec::new();
                for item in items {
                    nodes.push(AstNode::from_rule(&GrammarRule::Ref(item.clone()))?);
                }
                Ok(AstNode::Sequence(nodes))
            }
            GrammarRule::Alt(items) => {
                let mut nodes = Vec::new();
                for item in items {
                    nodes.push(AstNode::from_rule(&GrammarRule::Ref(item.clone()))?);
                }
                Ok(AstNode::Choice(nodes))
            }
            GrammarRule::Opt(rule_name) => {
                let node = AstNode::from_rule(&GrammarRule::Ref(rule_name.clone()))?;
                Ok(AstNode::Optional(Box::new(node)))
            }
            GrammarRule::Rep0(rule_name) => {
                let node = AstNode::from_rule(&GrammarRule::Ref(rule_name.clone()))?;
                Ok(AstNode::Repeat0(Box::new(node)))
            }
            GrammarRule::Rep1(rule_name) => {
                let node = AstNode::from_rule(&GrammarRule::Ref(rule_name.clone()))?;
                Ok(AstNode::Repeat1(Box::new(node)))
            }
            GrammarRule::Ref(name) => Ok(AstNode::Ref(name.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_node_nullable() {
        assert!(AstNode::Optional(Box::new(AstNode::Terminal("a".to_string()))).is_nullable());
        assert!(AstNode::Repeat0(Box::new(AstNode::Terminal("a".to_string()))).is_nullable());
        assert!(!AstNode::Terminal("a".to_string()).is_nullable());
    }

    #[test]
    fn test_ast_node_min_len() {
        assert_eq!(AstNode::Terminal("hello".to_string()).min_len(), 5);
        assert_eq!(AstNode::Class("a-z".to_string()).min_len(), 1);
        assert_eq!(
            AstNode::Optional(Box::new(AstNode::Terminal("a".to_string()))).min_len(),
            0
        );
        assert_eq!(
            AstNode::Repeat0(Box::new(AstNode::Terminal("a".to_string()))).min_len(),
            0
        );
        assert_eq!(
            AstNode::Repeat1(Box::new(AstNode::Terminal("a".to_string()))).min_len(),
            1
        );
    }

    #[test]
    fn test_ast_node_from_rule() {
        use super::super::GrammarRule;

        let rule = GrammarRule::Terminal("hello".to_string());
        let node = AstNode::from_rule(&rule).unwrap();
        assert_eq!(node, AstNode::Terminal("hello".to_string()));
    }
}
