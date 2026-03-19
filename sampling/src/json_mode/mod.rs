//! JSON mode for structured output
//!
//! Converts JSON schemas to GBNF grammars for constrained JSON generation.

use barq_core::grammar::GrammarCompiler;
use barq_core::grammar::{Grammar, GrammarError, GrammarParser, GrammarRule};
use std::collections::HashMap;
use vocab::Vocab;

/// JSON mode for constrained JSON output
pub struct JsonMode;

impl JsonMode {
    /// Convert JSON schema to GBNF grammar
    ///
    /// Takes a simplified JSON schema and returns GBNF grammar string
    pub fn schema_to_gbnf(schema: &JsonSchema) -> String {
        let mut gbnf = String::new();

        // Root rule
        gbnf.push_str(&format!("root ::= {}\n", schema.root_type()));

        // Add common rules
        gbnf.push_str(
            r#"
space ::= " "?
value ::= object | array | string | number | boolean | null
object ::= "{" space "}" | "{" space members "}"
members ::= member ("," space member)*
member ::= space string space ":" space value
array ::= "[" space "]" | "[" space elements "]"
elements ::= value ("," space value)*
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? digit+
digit ::= [0-9]
frac ::= "." digit+
exp ::= [eE] [+-]? digit+
boolean ::= "true" | "false"
null ::= "null"
"#,
        );

        gbnf
    }

    /// Create grammar sampler from JSON schema
    pub fn create_sampler(
        schema: &JsonSchema,
        vocab_size: usize,
    ) -> Result<crate::grammar_sampler::GrammarSampler, GrammarError> {
        let gbnf = Self::schema_to_gbnf(schema);
        crate::grammar_sampler::GrammarSamplerBuilder::new(vocab_size).from_gbnf(&gbnf)
    }

    /// Create a grammar sampler from JSON schema using the loaded vocabulary.
    pub fn create_sampler_with_vocab(
        schema: &JsonSchema,
        vocab: &Vocab,
    ) -> Result<crate::grammar_sampler::GrammarSampler, GrammarError> {
        let gbnf = Self::schema_to_gbnf(schema);
        crate::grammar_sampler::GrammarSamplerBuilder::new(vocab.len())
            .from_gbnf_with_vocab(&gbnf, vocab)
    }

    /// Validate generated JSON output and return the parsed value.
    pub fn validate_output(output: &str) -> Result<serde_json::Value, GrammarError> {
        serde_json::from_str(output.trim())
            .map_err(|err| GrammarError::InvalidJsonOutput(err.to_string()))
    }

    /// Create grammar for JSON object
    pub fn json_object_gbnf() -> String {
        r#"
root ::= object
object ::= "{" space "}" | "{" space members "}"
members ::= member ("," space member)*
member ::= space string space ":" space value
value ::= object | array | string | number | boolean | null
array ::= "[" space "]" | "[" space elements "]"
elements ::= value ("," space value)*
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? digit+
digit ::= [0-9]
frac ::= "." digit+
exp ::= [eE] [+-]? digit+
boolean ::= "true" | "false"
null ::= "null"
space ::= " "?
"#
        .to_string()
    }

    /// Create grammar for JSON array
    pub fn json_array_gbnf() -> String {
        r#"
root ::= array
array ::= "[" space "]" | "[" space elements "]"
elements ::= value ("," space value)*
value ::= object | array | string | number | boolean | null
object ::= "{" space "}" | "{" space members "}"
members ::= member ("," space member)*
member ::= space string space ":" space value
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? digit+
digit ::= [0-9]
frac ::= "." digit+
exp ::= [eE] [+-]? digit+
boolean ::= "true" | "false"
null ::= "null"
space ::= " "?
"#
        .to_string()
    }
}

/// JSON schema (simplified)
#[derive(Debug, Clone, PartialEq)]
pub enum JsonSchema {
    /// JSON object
    Object {
        properties: Vec<(String, JsonSchema)>,
        required: Vec<String>,
    },

    /// JSON array
    Array(Box<JsonSchema>),

    /// JSON string
    String,

    /// JSON number
    Number,

    /// JSON boolean
    Boolean,

    /// JSON null
    Null,

    /// One of multiple schemas
    OneOf(Vec<JsonSchema>),

    /// Any of multiple schemas
    AnyOf(Vec<JsonSchema>),
}

impl JsonSchema {
    /// Get root type name for GBNF
    fn root_type(&self) -> String {
        match self {
            JsonSchema::Object { .. } => "object".to_string(),
            JsonSchema::Array(_) => "array".to_string(),
            JsonSchema::String => "string".to_string(),
            JsonSchema::Number => "number".to_string(),
            JsonSchema::Boolean => "boolean".to_string(),
            JsonSchema::Null => "null".to_string(),
            JsonSchema::OneOf(schemas) | JsonSchema::AnyOf(schemas) => {
                let types: Vec<_> = schemas.iter().map(|s| s.root_type()).collect();
                types.join(" | ")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_object_gbnf() {
        let gbnf = JsonMode::json_object_gbnf();
        assert!(gbnf.contains("root ::= object"));
        assert!(gbnf.contains("object ::="));
        assert!(gbnf.contains("members ::="));
    }

    #[test]
    fn test_json_array_gbnf() {
        let gbnf = JsonMode::json_array_gbnf();
        assert!(gbnf.contains("root ::= array"));
        assert!(gbnf.contains("array ::="));
        assert!(gbnf.contains("elements ::="));
    }

    #[test]
    fn test_schema_to_gbnf() {
        let schema = JsonSchema::Object {
            properties: vec![("name".to_string(), JsonSchema::String)],
            required: vec!["name".to_string()],
        };

        let gbnf = JsonMode::schema_to_gbnf(&schema);
        assert!(gbnf.contains("root ::= object"));
    }

    #[test]
    fn test_json_root_types() {
        assert_eq!(JsonSchema::String.root_type(), "string");
        assert_eq!(JsonSchema::Number.root_type(), "number");
        assert_eq!(JsonSchema::Boolean.root_type(), "boolean");
        assert_eq!(JsonSchema::Null.root_type(), "null");
    }

    #[test]
    fn test_validate_output() {
        let parsed = JsonMode::validate_output(r#"{"name":"barq"}"#).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn test_create_sampler_with_vocab() {
        use vocab::{Token, TokenAttr, TokenType, Vocab, VocabType};

        let mut vocab = Vocab::new(VocabType::SPM);
        vocab.add_token(Token {
            id: 0,
            text: "{".to_string(),
            score: 0.0,
            token_type: TokenType::Normal,
            attrs: TokenAttr::default(),
        });
        vocab.add_token(Token {
            id: 1,
            text: "}".to_string(),
            score: 0.0,
            token_type: TokenType::Normal,
            attrs: TokenAttr::default(),
        });

        let schema = JsonSchema::Object {
            properties: vec![],
            required: vec![],
        };

        let sampler = JsonMode::create_sampler_with_vocab(&schema, &vocab).unwrap();
        assert!(!sampler.get_allowed_tokens().is_empty());
    }
}
