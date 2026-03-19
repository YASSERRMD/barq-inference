//! Chat template rendering for chat-oriented models.
//!
//! This implements a small Jinja2-style subset that covers the prompt patterns
//! used by the common model families in this repository:
//! - variable interpolation with `{{ ... }}`
//! - conditional blocks with `{% if ... %}` / `{% else %}` / `{% endif %}`
//! - message loops with `{% for message in messages %}` / `{% endfor %}`
//!
//! The goal is to render chat prompts deterministically from model metadata and
//! token vocabularies without depending on a heavyweight template engine.

use crate::vocab::Vocab;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported chat roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Developer,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Developer => "developer",
        }
    }
}

impl fmt::Display for ChatRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Built-in chat template families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatTemplatePreset {
    Llama,
    Mistral,
    Qwen,
    ChatML,
    Generic,
}

impl ChatTemplatePreset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Qwen => "qwen",
            Self::ChatML => "chatml",
            Self::Generic => "generic",
        }
    }

    fn default_markers(self) -> RoleMarkers {
        match self {
            Self::Llama | Self::Mistral => RoleMarkers {
                system_prefix: "<<SYS>>\n".to_string(),
                system_suffix: "\n<</SYS>>\n\n".to_string(),
                user_prefix: "[INST] ".to_string(),
                user_suffix: " [/INST]\n".to_string(),
                assistant_prefix: String::new(),
                assistant_suffix: "</s>\n".to_string(),
                tool_prefix: "[TOOL] ".to_string(),
                tool_suffix: "\n".to_string(),
                developer_prefix: "[DEV] ".to_string(),
                developer_suffix: "\n".to_string(),
            },
            Self::Qwen | Self::ChatML => RoleMarkers {
                system_prefix: "<|im_start|>system\n".to_string(),
                system_suffix: "<|im_end|>\n".to_string(),
                user_prefix: "<|im_start|>user\n".to_string(),
                user_suffix: "<|im_end|>\n".to_string(),
                assistant_prefix: "<|im_start|>assistant\n".to_string(),
                assistant_suffix: "<|im_end|>\n".to_string(),
                tool_prefix: "<|im_start|>tool\n".to_string(),
                tool_suffix: "<|im_end|>\n".to_string(),
                developer_prefix: "<|im_start|>developer\n".to_string(),
                developer_suffix: "<|im_end|>\n".to_string(),
            },
            Self::Generic => RoleMarkers {
                system_prefix: "[SYSTEM] ".to_string(),
                system_suffix: "\n".to_string(),
                user_prefix: "[USER] ".to_string(),
                user_suffix: "\n".to_string(),
                assistant_prefix: "[ASSISTANT] ".to_string(),
                assistant_suffix: "\n".to_string(),
                tool_prefix: "[TOOL] ".to_string(),
                tool_suffix: "\n".to_string(),
                developer_prefix: "[DEVELOPER] ".to_string(),
                developer_suffix: "\n".to_string(),
            },
        }
    }
}

/// Template-specific role markers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleMarkers {
    pub system_prefix: String,
    pub system_suffix: String,
    pub user_prefix: String,
    pub user_suffix: String,
    pub assistant_prefix: String,
    pub assistant_suffix: String,
    pub tool_prefix: String,
    pub tool_suffix: String,
    pub developer_prefix: String,
    pub developer_suffix: String,
}

impl RoleMarkers {
    fn prefix_for(&self, role: ChatRole) -> &str {
        match role {
            ChatRole::System => &self.system_prefix,
            ChatRole::User => &self.user_prefix,
            ChatRole::Assistant => &self.assistant_prefix,
            ChatRole::Tool => &self.tool_prefix,
            ChatRole::Developer => &self.developer_prefix,
        }
    }

    fn suffix_for(&self, role: ChatRole) -> &str {
        match role {
            ChatRole::System => &self.system_suffix,
            ChatRole::User => &self.user_suffix,
            ChatRole::Assistant => &self.assistant_suffix,
            ChatRole::Tool => &self.tool_suffix,
            ChatRole::Developer => &self.developer_suffix,
        }
    }
}

#[derive(Debug, Clone)]
struct RenderTokens {
    bos: String,
    eos: String,
    newline: String,
}

impl RenderTokens {
    fn from_vocab(vocab: Option<&Vocab>) -> Self {
        let bos = vocab
            .and_then(|v| v.special_tokens().bos.and_then(|id| v.get_token(id)))
            .map(|token| token.text.clone())
            .unwrap_or_else(|| "<s>".to_string());
        let eos = vocab
            .and_then(|v| v.special_tokens().eos.and_then(|id| v.get_token(id)))
            .map(|token| token.text.clone())
            .unwrap_or_else(|| "</s>".to_string());
        let newline = vocab
            .and_then(|v| v.special_tokens().nl.and_then(|id| v.get_token(id)))
            .map(|token| token.text.clone())
            .unwrap_or_else(|| "\n".to_string());

        Self { bos, eos, newline }
    }
}

/// Jinja-like chat template.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    preset: ChatTemplatePreset,
    name: String,
    source: String,
    markers: RoleMarkers,
}

impl ChatTemplate {
    /// Create a built-in template for the requested preset.
    pub fn preset(preset: ChatTemplatePreset) -> Self {
        let (name, source) = match preset {
            ChatTemplatePreset::Llama => (
                "llama",
                "{{ bos_token }}{% if system_prompt %}{{ system_prefix }}{{ system_prompt }}{{ system_suffix }}{% endif %}{% for message in messages %}{{ message.role_prefix }}{{ message.content }}{{ message.role_suffix }}{% endfor %}{{ eos_token }}",
            ),
            ChatTemplatePreset::Mistral => (
                "mistral",
                "{{ bos_token }}{% if system_prompt %}{{ system_prefix }}{{ system_prompt }}{{ system_suffix }}{% endif %}{% for message in messages %}{{ message.role_prefix }}{{ message.content }}{{ message.role_suffix }}{% endfor %}{{ eos_token }}",
            ),
            ChatTemplatePreset::Qwen => (
                "qwen",
                "{{ bos_token }}{% if system_prompt %}{{ system_prefix }}{{ system_prompt }}{{ system_suffix }}{% endif %}{% for message in messages %}{{ message.role_prefix }}{{ message.content }}{{ message.role_suffix }}{% endfor %}{{ eos_token }}",
            ),
            ChatTemplatePreset::ChatML => (
                "chatml",
                "{{ bos_token }}{% if system_prompt %}{{ system_prefix }}{{ system_prompt }}{{ system_suffix }}{% endif %}{% for message in messages %}{{ message.role_prefix }}{{ message.content }}{{ message.role_suffix }}{% endfor %}{{ eos_token }}",
            ),
            ChatTemplatePreset::Generic => (
                "generic",
                "{{ bos_token }}{% if system_prompt %}{{ system_prefix }}{{ system_prompt }}{{ system_suffix }}{% endif %}{% for message in messages %}{{ message.role_prefix }}{{ message.content }}{{ message.role_suffix }}{% endfor %}{{ eos_token }}",
            ),
        };

        Self {
            preset,
            name: name.to_string(),
            source: source.to_string(),
            markers: preset.default_markers(),
        }
    }

    /// Build a template from an architecture name.
    pub fn for_arch(arch_name: &str) -> Self {
        let normalized = arch_name.to_ascii_lowercase();
        let preset = match normalized.as_str() {
            "llama" | "llama2" | "llama3" | "mistral" | "mixtral" => ChatTemplatePreset::Llama,
            "qwen" | "qwen2" | "qwen3" => ChatTemplatePreset::Qwen,
            "chatml" | "deepseek" | "deepseek.moe" => ChatTemplatePreset::ChatML,
            _ => ChatTemplatePreset::Generic,
        };
        Self::preset(preset)
    }

    /// Create a custom template.
    pub fn custom(
        name: impl Into<String>,
        source: impl Into<String>,
        markers: RoleMarkers,
    ) -> Self {
        Self {
            preset: ChatTemplatePreset::Generic,
            name: name.into(),
            source: source.into(),
            markers,
        }
    }

    /// Returns the template preset.
    pub fn preset_kind(&self) -> ChatTemplatePreset {
        self.preset
    }

    /// Returns the template name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the raw template source.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render a prompt using the template.
    pub fn render(
        &self,
        vocab: Option<&Vocab>,
        system_prompt: Option<&str>,
        messages: &[ChatMessage],
    ) -> Result<String> {
        let tokens = RenderTokens::from_vocab(vocab);
        let nodes = parse_template(&self.source)?;
        let mut ctx = RenderContext {
            system_prompt,
            messages,
            current_message: None,
            markers: &self.markers,
            tokens,
        };

        let mut rendered = String::new();
        render_nodes(&nodes, &mut ctx, &mut rendered)?;
        Ok(rendered)
    }
}

#[derive(Debug, Clone)]
struct RenderContext<'a> {
    system_prompt: Option<&'a str>,
    messages: &'a [ChatMessage],
    current_message: Option<&'a ChatMessage>,
    markers: &'a RoleMarkers,
    tokens: RenderTokens,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Text(String),
    Var(String),
    If {
        condition: String,
        then_branch: Vec<Node>,
        else_branch: Vec<Node>,
    },
    For {
        item_name: String,
        list_name: String,
        body: Vec<Node>,
    },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ChatTemplateError {
    #[error("template syntax error: {0}")]
    Syntax(String),
    #[error("template references unknown variable: {0}")]
    UnknownVariable(String),
    #[error("template control flow error: {0}")]
    ControlFlow(String),
}

type Result<T> = std::result::Result<T, ChatTemplateError>;

fn parse_template(source: &str) -> Result<Vec<Node>> {
    let mut pos = 0usize;
    let (nodes, terminator) = parse_nodes(source, &mut pos, &[])?;
    if terminator.is_some() {
        return Err(ChatTemplateError::Syntax(
            "unexpected terminator at top level".to_string(),
        ));
    }
    if pos != source.len() {
        return Err(ChatTemplateError::Syntax(
            "template parser stopped early".to_string(),
        ));
    }
    Ok(nodes)
}

fn parse_nodes(
    source: &str,
    pos: &mut usize,
    stop_tags: &[&str],
) -> Result<(Vec<Node>, Option<String>)> {
    let mut nodes = Vec::new();

    while *pos < source.len() {
        if let Some(rest) = source.get(*pos..) {
            if rest.starts_with("{{") {
                let end = rest.find("}}").ok_or_else(|| {
                    ChatTemplateError::Syntax("unterminated variable tag".to_string())
                })?;
                let expr = rest[2..end].trim();
                nodes.push(Node::Var(expr.to_string()));
                *pos += end + 2;
                continue;
            }

            if rest.starts_with("{%") {
                let end = rest.find("%}").ok_or_else(|| {
                    ChatTemplateError::Syntax("unterminated block tag".to_string())
                })?;
                let tag = rest[2..end].trim().to_string();
                *pos += end + 2;

                let tag_name = tag.split_whitespace().next().unwrap_or("");
                if stop_tags.contains(&tag_name) {
                    return Ok((nodes, Some(tag_name.to_string())));
                }

                match tag_name {
                    "if" => {
                        let condition = tag[2..].trim().to_string();
                        let (then_branch, terminator) =
                            parse_nodes(source, pos, &["else", "endif"])?;
                        let else_branch = if matches!(terminator.as_deref(), Some("else")) {
                            let (else_branch, end_terminator) =
                                parse_nodes(source, pos, &["endif"])?;
                            if !matches!(end_terminator.as_deref(), Some("endif")) {
                                return Err(ChatTemplateError::Syntax(
                                    "missing endif after else branch".to_string(),
                                ));
                            }
                            else_branch
                        } else {
                            if !matches!(terminator.as_deref(), Some("endif")) {
                                return Err(ChatTemplateError::Syntax(
                                    "missing endif for conditional".to_string(),
                                ));
                            }
                            Vec::new()
                        };

                        nodes.push(Node::If {
                            condition,
                            then_branch,
                            else_branch,
                        });
                    }
                    "for" => {
                        let expr = tag[3..].trim();
                        let mut parts = expr.split_whitespace();
                        let item_name = parts.next().ok_or_else(|| {
                            ChatTemplateError::Syntax("for loop missing item name".to_string())
                        })?;
                        let in_kw = parts.next().ok_or_else(|| {
                            ChatTemplateError::Syntax("for loop missing `in`".to_string())
                        })?;
                        let list_name = parts.next().ok_or_else(|| {
                            ChatTemplateError::Syntax("for loop missing list name".to_string())
                        })?;
                        if in_kw != "in" {
                            return Err(ChatTemplateError::Syntax(
                                "for loop must use `in`".to_string(),
                            ));
                        }

                        let (body, terminator) = parse_nodes(source, pos, &["endfor"])?;
                        if !matches!(terminator.as_deref(), Some("endfor")) {
                            return Err(ChatTemplateError::Syntax(
                                "missing endfor for loop".to_string(),
                            ));
                        }

                        nodes.push(Node::For {
                            item_name: item_name.to_string(),
                            list_name: list_name.to_string(),
                            body,
                        });
                    }
                    "else" | "endif" | "endfor" => {
                        return Ok((nodes, Some(tag_name.to_string())));
                    }
                    other => {
                        return Err(ChatTemplateError::Syntax(format!(
                            "unsupported template tag `{}`",
                            other
                        )));
                    }
                }

                continue;
            }

            let next_var = rest.find("{{");
            let next_block = rest.find("{%");
            let next = match (next_var, next_block) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => rest.len(),
            };

            if next == 0 {
                return Err(ChatTemplateError::Syntax(
                    "parser made no progress".to_string(),
                ));
            }

            nodes.push(Node::Text(rest[..next].to_string()));
            *pos += next;
        }
    }

    Ok((nodes, None))
}

fn render_nodes(nodes: &[Node], ctx: &mut RenderContext<'_>, out: &mut String) -> Result<()> {
    for node in nodes {
        match node {
            Node::Text(text) => out.push_str(text),
            Node::Var(expr) => out.push_str(&resolve_expr(expr, ctx)?),
            Node::If {
                condition,
                then_branch,
                else_branch,
            } => {
                if resolve_bool(condition, ctx)? {
                    render_nodes(then_branch, ctx, out)?;
                } else {
                    render_nodes(else_branch, ctx, out)?;
                }
            }
            Node::For {
                item_name,
                list_name,
                body,
            } => {
                if item_name != "message" {
                    return Err(ChatTemplateError::Syntax(format!(
                        "unsupported loop variable `{}`",
                        item_name
                    )));
                }

                if list_name != "messages" {
                    return Err(ChatTemplateError::Syntax(format!(
                        "unsupported loop source `{}`",
                        list_name
                    )));
                }

                for message in ctx.messages {
                    let previous = ctx.current_message;
                    ctx.current_message = Some(message);
                    render_nodes(body, ctx, out)?;
                    ctx.current_message = previous;
                }
            }
        }
    }

    Ok(())
}

fn resolve_expr(expr: &str, ctx: &RenderContext<'_>) -> Result<String> {
    match expr {
        "system_prompt" => Ok(ctx.system_prompt.unwrap_or("").to_string()),
        "bos_token" => Ok(ctx.tokens.bos.clone()),
        "eos_token" => Ok(ctx.tokens.eos.clone()),
        "newline" | "nl_token" => Ok(ctx.tokens.newline.clone()),
        "system_prefix" => Ok(ctx.markers.system_prefix.clone()),
        "system_suffix" => Ok(ctx.markers.system_suffix.clone()),
        "message.role" => Ok(ctx
            .current_message
            .map(|message| message.role.as_str().to_string())
            .unwrap_or_default()),
        "message.content" => Ok(ctx
            .current_message
            .map(|message| message.content.clone())
            .unwrap_or_default()),
        "message.role_prefix" => Ok(ctx
            .current_message
            .map(|message| ctx.markers.prefix_for(message.role).to_string())
            .unwrap_or_default()),
        "message.role_suffix" => Ok(ctx
            .current_message
            .map(|message| ctx.markers.suffix_for(message.role).to_string())
            .unwrap_or_default()),
        "message.is_system" => Ok(bool_string(
            ctx.current_message
                .is_some_and(|m| m.role == ChatRole::System),
        )),
        "message.is_user" => Ok(bool_string(
            ctx.current_message
                .is_some_and(|m| m.role == ChatRole::User),
        )),
        "message.is_assistant" => Ok(bool_string(
            ctx.current_message
                .is_some_and(|m| m.role == ChatRole::Assistant),
        )),
        other => Err(ChatTemplateError::UnknownVariable(other.to_string())),
    }
}

fn resolve_bool(expr: &str, ctx: &RenderContext<'_>) -> Result<bool> {
    match expr {
        "system_prompt" => Ok(ctx
            .system_prompt
            .map(|prompt| !prompt.is_empty())
            .unwrap_or(false)),
        "messages" => Ok(!ctx.messages.is_empty()),
        "message.is_system" => Ok(ctx
            .current_message
            .is_some_and(|message| message.role == ChatRole::System)),
        "message.is_user" => Ok(ctx
            .current_message
            .is_some_and(|message| message.role == ChatRole::User)),
        "message.is_assistant" => Ok(ctx
            .current_message
            .is_some_and(|message| message.role == ChatRole::Assistant)),
        other => Err(ChatTemplateError::UnknownVariable(other.to_string())),
    }
}

fn bool_string(value: bool) -> String {
    if value {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vocab() -> Vocab {
        use crate::vocab::{SpecialToken, Token, TokenAttr, TokenType, VocabType};

        let mut vocab = Vocab::new(VocabType::SPM);
        vocab.add_token(Token {
            id: 1,
            text: "<s>".to_string(),
            score: 0.0,
            token_type: TokenType::Control,
            attrs: TokenAttr::default(),
        });
        vocab.add_token(Token {
            id: 2,
            text: "</s>".to_string(),
            score: 0.0,
            token_type: TokenType::Control,
            attrs: TokenAttr::default(),
        });
        vocab.add_token(Token {
            id: 3,
            text: "\n".to_string(),
            score: 0.0,
            token_type: TokenType::Control,
            attrs: TokenAttr::default(),
        });
        vocab.set_special_tokens(SpecialToken {
            bos: Some(1),
            eos: Some(2),
            nl: Some(3),
            ..Default::default()
        });
        vocab
    }

    #[test]
    fn test_template_parser_if_and_for() {
        let template = ChatTemplate::custom(
            "custom",
            "{{ bos_token }}{% if system_prompt %}[SYS]{{ system_prompt }}{% endif %}{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}{{ eos_token }}",
            ChatTemplatePreset::Generic.default_markers(),
        );
        let rendered = template
            .render(
                Some(&test_vocab()),
                Some("You are helpful"),
                &[
                    ChatMessage::new(ChatRole::User, "Hello"),
                    ChatMessage::new(ChatRole::Assistant, "Hi"),
                ],
            )
            .unwrap();

        assert!(rendered.contains("[SYS]You are helpful"));
        assert!(rendered.contains("user: Hello"));
        assert!(rendered.contains("assistant: Hi"));
        assert!(rendered.starts_with("<s>"));
        assert!(rendered.ends_with("</s>"));
    }

    #[test]
    fn test_llama_template_rendering() {
        let template = ChatTemplate::preset(ChatTemplatePreset::Llama);
        let rendered = template
            .render(
                Some(&test_vocab()),
                Some("Be concise"),
                &[
                    ChatMessage::new(ChatRole::User, "What is Rust?"),
                    ChatMessage::new(ChatRole::Assistant, "A systems language."),
                ],
            )
            .unwrap();

        assert!(rendered.contains("Be concise"));
        assert!(rendered.contains("What is Rust?"));
        assert!(rendered.contains("A systems language."));
        assert!(rendered.contains("[INST]"));
    }

    #[test]
    fn test_template_selection_from_arch() {
        assert_eq!(
            ChatTemplate::for_arch("qwen2").preset_kind(),
            ChatTemplatePreset::Qwen
        );
        assert_eq!(
            ChatTemplate::for_arch("mistral").preset_kind(),
            ChatTemplatePreset::Llama
        );
        assert_eq!(
            ChatTemplate::for_arch("unknown").preset_kind(),
            ChatTemplatePreset::Generic
        );
    }
}
