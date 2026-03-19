//! Architecture registry
//!
//! Centralized registry for mapping GGUF architecture names to implementations.
//! Provides automatic architecture detection and model instantiation.

use crate::arch::LlmArch;
use crate::deepseek::{DeepSeekMoEModel, DeepSeekModel};
use crate::llama::LlamaModel;
use crate::llava::LlavaModel;
use crate::loader::Model;
use crate::mistral::MistralModel;
use crate::mixtral::MixtralModel;
use crate::qwen::QwenModel;
use crate::qwen2::{Qwen2MoEModel, Qwen2Model};
use crate::qwen2vl::Qwen2VlModel;
use crate::qwen3::Qwen3Model;
use barq_core::error::{Error, Result};
use std::sync::Arc;

/// Unified model trait for all architecture implementations
pub trait LlmArchTrait: Send + Sync {
    /// Get architecture type
    fn arch(&self) -> LlmArch;

    /// Get architecture name
    fn arch_name(&self) -> &'static str {
        self.arch().name()
    }

    /// Get model reference
    fn model(&self) -> &Model;

    /// Check if architecture supports GQA (Grouped Query Attention)
    fn supports_gqa(&self) -> bool {
        false
    }

    /// Check if architecture supports MoE (Mixture of Experts)
    fn supports_moe(&self) -> bool {
        false
    }

    /// Check if architecture uses sliding window attention
    fn supports_sliding_window(&self) -> bool {
        false
    }

    /// Check if architecture supports RoPE scaling
    fn supports_rope_scaling(&self) -> bool {
        false
    }

    /// Check if architecture supports vision inputs
    fn supports_vision(&self) -> bool {
        false
    }

    /// Get description of architecture features
    fn features(&self) -> Vec<&'static str> {
        vec![]
    }
}

// Implement LlmArchTrait for each model type
impl LlmArchTrait for LlamaModel {
    fn arch(&self) -> LlmArch {
        LlmArch::Llama
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_gqa(&self) -> bool {
        self.inner().hparams.n_head_kv < self.inner().hparams.n_head
    }

    fn features(&self) -> Vec<&'static str> {
        let mut features = vec!["RoPE"];
        if self.supports_gqa() {
            features.push("GQA");
        }
        features
    }
}

impl LlmArchTrait for MistralModel {
    fn arch(&self) -> LlmArch {
        LlmArch::Mistral
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_gqa(&self) -> bool {
        self.has_gqa()
    }

    fn supports_sliding_window(&self) -> bool {
        self.has_sliding_window()
    }

    fn supports_rope_scaling(&self) -> bool {
        self.has_rope_scaling()
    }

    fn features(&self) -> Vec<&'static str> {
        let mut features = vec!["RoPE"];
        if self.supports_gqa() {
            features.push("GQA");
        }
        if self.supports_sliding_window() {
            features.push("Sliding Window");
        }
        if self.supports_rope_scaling() {
            features.push("RoPE Scaling");
        }
        features
    }
}

impl LlmArchTrait for MixtralModel {
    fn arch(&self) -> LlmArch {
        LlmArch::Mixtral
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_gqa(&self) -> bool {
        self.has_gqa()
    }

    fn supports_moe(&self) -> bool {
        true
    }

    fn supports_sliding_window(&self) -> bool {
        self.has_sliding_window()
    }

    fn supports_rope_scaling(&self) -> bool {
        self.has_rope_scaling()
    }

    fn features(&self) -> Vec<&'static str> {
        let mut features = vec!["MoE", "RoPE"];
        if self.supports_gqa() {
            features.push("GQA");
        }
        if self.supports_sliding_window() {
            features.push("Sliding Window");
        }
        if self.has_load_balancing() {
            features.push("Load Balancing");
        }
        if self.supports_rope_scaling() {
            features.push("RoPE Scaling");
        }
        features
    }
}

impl LlmArchTrait for QwenModel {
    fn arch(&self) -> LlmArch {
        LlmArch::Qwen
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_rope_scaling(&self) -> bool {
        self.has_ntk_scaling()
    }

    fn features(&self) -> Vec<&'static str> {
        vec!["NTK-aware RoPE"]
    }
}

impl LlmArchTrait for Qwen2Model {
    fn arch(&self) -> LlmArch {
        LlmArch::Qwen2
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_gqa(&self) -> bool {
        self.has_gqa()
    }

    fn supports_rope_scaling(&self) -> bool {
        self.has_ntk_scaling()
    }

    fn features(&self) -> Vec<&'static str> {
        let mut features = vec!["NTK-aware RoPE"];
        if self.supports_gqa() {
            features.push("GQA");
        }
        features
    }
}

impl LlmArchTrait for Qwen2MoEModel {
    fn arch(&self) -> LlmArch {
        LlmArch::Qwen2Moe
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_gqa(&self) -> bool {
        self.has_gqa()
    }

    fn supports_moe(&self) -> bool {
        true
    }

    fn supports_rope_scaling(&self) -> bool {
        self.inner().hparams.rope_scaling_type > 0
    }

    fn features(&self) -> Vec<&'static str> {
        let mut features = vec!["MoE", "NTK-aware RoPE"];
        if self.supports_gqa() {
            features.push("GQA");
        }
        features
    }
}

impl LlmArchTrait for Qwen3Model {
    fn arch(&self) -> LlmArch {
        LlmArch::Qwen3
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_gqa(&self) -> bool {
        self.has_gqa()
    }

    fn supports_rope_scaling(&self) -> bool {
        self.has_ntk_scaling()
    }

    fn features(&self) -> Vec<&'static str> {
        let mut features = vec!["NTK-aware RoPE"];
        if self.supports_gqa() {
            features.push("GQA");
        }
        features
    }
}

impl LlmArchTrait for DeepSeekModel {
    fn arch(&self) -> LlmArch {
        LlmArch::DeepSeek
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_rope_scaling(&self) -> bool {
        self.has_yarn_scaling()
    }

    fn features(&self) -> Vec<&'static str> {
        vec!["MLA", "SwiGLU", "Yarn RoPE"]
    }
}

impl LlmArchTrait for DeepSeekMoEModel {
    fn arch(&self) -> LlmArch {
        LlmArch::DeepSeekMoE
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_moe(&self) -> bool {
        true
    }

    fn supports_rope_scaling(&self) -> bool {
        self.inner().hparams.rope_scaling_type == 2
    }

    fn features(&self) -> Vec<&'static str> {
        vec!["MoE", "MLA", "SwiGLU", "Load Balancing"]
    }
}

impl LlmArchTrait for Qwen2VlModel {
    fn arch(&self) -> LlmArch {
        LlmArch::Qwen2Vl
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn features(&self) -> Vec<&'static str> {
        vec!["Vision", "Cross-Attention", "Image Tokens"]
    }
}

impl LlmArchTrait for LlavaModel {
    fn arch(&self) -> LlmArch {
        LlmArch::Llava
    }

    fn model(&self) -> &Model {
        self.inner()
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn features(&self) -> Vec<&'static str> {
        vec!["Vision", "Cross-Attention", "Image Tokens"]
    }
}

/// Architecture registry for automatic model instantiation
pub struct ArchitectureRegistry;

impl ArchitectureRegistry {
    /// Create appropriate model wrapper based on architecture
    pub fn create_model(model: Arc<Model>) -> Result<Box<dyn LlmArchTrait>> {
        let arch = model.arch();

        match arch {
            LlmArch::Llama => Ok(Box::new(LlamaModel::new(model)?)),
            LlmArch::Mistral => Ok(Box::new(MistralModel::new(model)?)),
            LlmArch::Mixtral => Ok(Box::new(MixtralModel::new(model)?)),
            LlmArch::Qwen => Ok(Box::new(QwenModel::new(model)?)),
            LlmArch::Qwen2 => Ok(Box::new(Qwen2Model::new(model)?)),
            LlmArch::Qwen2Moe => Ok(Box::new(Qwen2MoEModel::new(model)?)),
            LlmArch::Qwen3 => Ok(Box::new(Qwen3Model::new(model)?)),
            LlmArch::Qwen2Vl => Ok(Box::new(Qwen2VlModel::new(model)?)),
            LlmArch::Llava => Ok(Box::new(LlavaModel::new(model)?)),
            LlmArch::DeepSeek => Ok(Box::new(DeepSeekModel::new(model)?)),
            LlmArch::DeepSeekMoE => Ok(Box::new(DeepSeekMoEModel::new(model)?)),
            _ => Err(Error::Unsupported(format!(
                "Architecture {:?} not yet implemented",
                arch
            ))),
        }
    }

    /// Get list of supported architectures
    pub fn supported_architectures() -> Vec<LlmArch> {
        vec![
            LlmArch::Llama,
            LlmArch::Mistral,
            LlmArch::Mixtral,
            LlmArch::Qwen,
            LlmArch::Qwen2,
            LlmArch::Qwen2Moe,
            LlmArch::Qwen3,
            LlmArch::Qwen2Vl,
            LlmArch::Llava,
            LlmArch::DeepSeek,
            LlmArch::DeepSeekMoE,
        ]
    }

    /// Check if architecture is supported
    pub fn is_supported(arch: LlmArch) -> bool {
        Self::supported_architectures().contains(&arch)
    }

    /// Get architecture from GGUF architecture name string
    pub fn from_name(name: &str) -> Option<LlmArch> {
        match name.to_lowercase().as_str() {
            "llama" => Some(LlmArch::Llama),
            "mistral" => Some(LlmArch::Mistral),
            "mixtral" => Some(LlmArch::Mixtral),
            "qwen" => Some(LlmArch::Qwen),
            "qwen2" => Some(LlmArch::Qwen2),
            "qwen2.moe" | "qwen2moe" => Some(LlmArch::Qwen2Moe),
            "qwen3" => Some(LlmArch::Qwen3),
            "qwen2vl" | "qwen2.vl" | "qwen-vl" => Some(LlmArch::Qwen2Vl),
            "llava" | "llava-1.5" | "llava-1.6" => Some(LlmArch::Llava),
            "deepseek" => Some(LlmArch::DeepSeek),
            "deepseek.moe" | "deepseekmoe" => Some(LlmArch::DeepSeekMoE),
            _ => None,
        }
    }

    /// Get architecture description
    pub fn describe(arch: LlmArch) -> &'static str {
        match arch {
            LlmArch::Llama => "LLaMA - Large Language Model Meta AI",
            LlmArch::Mistral => "Mistral 7B - GQA, Sliding Window, RoPE",
            LlmArch::Mixtral => "Mixtral 8x7B/8x22B - MoE, GQA, Sliding Window",
            LlmArch::Qwen => "Qwen - Alibaba Cloud, NTK-aware RoPE",
            LlmArch::Qwen2 => "Qwen2 - GQA, improved RoPE",
            LlmArch::Qwen2Moe => "Qwen2-MoE - Mixture of Experts variant",
            LlmArch::Qwen3 => "Qwen3 - GQA, NTK-aware RoPE",
            LlmArch::DeepSeek => "DeepSeek - MLA (Multi-head Latent Attention)",
            LlmArch::DeepSeekMoE => "DeepSeek-MoE - MLA with sparse experts",
            _ => "Unknown architecture",
        }
    }

    /// Get architecture capabilities
    pub fn capabilities(arch: LlmArch) -> Vec<&'static str> {
        match arch {
            LlmArch::Llama => vec!["RoPE", "GQA"],
            LlmArch::Mistral => vec!["RoPE", "GQA", "Sliding Window"],
            LlmArch::Mixtral => vec!["MoE", "RoPE", "GQA", "Sliding Window", "Load Balancing"],
            LlmArch::Qwen => vec!["NTK-aware RoPE"],
            LlmArch::Qwen2 => vec!["GQA", "NTK-aware RoPE"],
            LlmArch::Qwen2Moe => vec!["MoE", "GQA", "NTK-aware RoPE"],
            LlmArch::Qwen3 => vec!["GQA", "NTK-aware RoPE"],
            LlmArch::DeepSeek => vec!["MLA", "SwiGLU", "Yarn RoPE"],
            LlmArch::DeepSeekMoE => vec!["MoE", "MLA", "SwiGLU", "Load Balancing"],
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_test_gguf_file;
    use barq_core::gguf::GgufValue;
    use std::sync::Arc;

    async fn load_fixture_model(prefix: &str, kv_pairs: &[(&str, GgufValue)]) -> Arc<Model> {
        let path = write_test_gguf_file(prefix, kv_pairs);
        Arc::new(Model::load(&path).await.unwrap())
    }

    #[test]
    fn test_architecture_from_name() {
        assert_eq!(
            ArchitectureRegistry::from_name("llama"),
            Some(LlmArch::Llama)
        );
        assert_eq!(
            ArchitectureRegistry::from_name("Mistral"),
            Some(LlmArch::Mistral)
        );
        assert_eq!(
            ArchitectureRegistry::from_name("qwen2.moe"),
            Some(LlmArch::Qwen2Moe)
        );
        assert_eq!(
            ArchitectureRegistry::from_name("qwen3"),
            Some(LlmArch::Qwen3)
        );
        assert_eq!(
            ArchitectureRegistry::from_name("deepseek"),
            Some(LlmArch::DeepSeek)
        );
        assert_eq!(
            ArchitectureRegistry::from_name("deepseek.moe"),
            Some(LlmArch::DeepSeekMoE)
        );
        assert_eq!(
            ArchitectureRegistry::from_name("qwen2.vl"),
            Some(LlmArch::Qwen2Vl)
        );
        assert_eq!(
            ArchitectureRegistry::from_name("llava-1.6"),
            Some(LlmArch::Llava)
        );
        assert_eq!(ArchitectureRegistry::from_name("unknown"), None);
    }

    #[test]
    fn test_supported_architectures() {
        let supported = ArchitectureRegistry::supported_architectures();
        assert!(supported.contains(&LlmArch::Llama));
        assert!(supported.contains(&LlmArch::Mistral));
        assert!(supported.contains(&LlmArch::Mixtral));
        assert!(supported.contains(&LlmArch::Qwen));
        assert!(supported.contains(&LlmArch::Qwen2));
        assert!(supported.contains(&LlmArch::Qwen3));
        assert!(supported.contains(&LlmArch::Qwen2Vl));
        assert!(supported.contains(&LlmArch::Llava));
        assert!(supported.contains(&LlmArch::DeepSeek));
        assert!(supported.contains(&LlmArch::DeepSeekMoE));
    }

    #[test]
    fn test_architecture_capabilities() {
        let caps = ArchitectureRegistry::capabilities(LlmArch::Mixtral);
        assert!(caps.contains(&"MoE"));
        assert!(caps.contains(&"GQA"));
        assert!(caps.contains(&"Sliding Window"));
    }

    #[test]
    fn test_qwen3_capabilities() {
        let caps = ArchitectureRegistry::capabilities(LlmArch::Qwen3);
        assert!(caps.contains(&"GQA"));
        assert!(caps.contains(&"NTK-aware RoPE"));
    }

    #[tokio::test]
    async fn test_create_model_deepseek() {
        let model = load_fixture_model(
            "arch-registry-deepseek",
            &[
                (
                    "general.architecture",
                    GgufValue::String("deepseek".to_string()),
                ),
                ("deepseek.block_count", GgufValue::Uint32(28)),
                ("deepseek.attention.head_count", GgufValue::Uint32(32)),
                ("deepseek.attention.head_count_kv", GgufValue::Uint32(8)),
                ("deepseek.embedding_length", GgufValue::Uint32(4096)),
                ("deepseek.intermediate_size", GgufValue::Uint32(11_008)),
                ("deepseek.context_length", GgufValue::Uint32(131_072)),
                ("deepseek.rope.scaling.type", GgufValue::Uint32(2)),
                ("deepseek.n_latent_heads", GgufValue::Uint32(8)),
            ],
        )
        .await;

        let arch = ArchitectureRegistry::create_model(model).unwrap();
        assert_eq!(arch.arch(), LlmArch::DeepSeek);
        assert_eq!(arch.arch_name(), "deepseek");
        assert!(arch.features().contains(&"MLA"));
        assert!(arch.features().contains(&"Yarn RoPE"));
    }

    #[tokio::test]
    async fn test_create_model_mixtral() {
        let model = load_fixture_model(
            "arch-registry-mixtral",
            &[
                (
                    "general.architecture",
                    GgufValue::String("mixtral".to_string()),
                ),
                ("llama.block_count", GgufValue::Uint32(32)),
                ("llama.attention.head_count", GgufValue::Uint32(32)),
                ("llama.attention.head_count_kv", GgufValue::Uint32(8)),
                ("llama.embedding_length", GgufValue::Uint32(4096)),
                ("llama.feed_forward_length", GgufValue::Uint32(14_336)),
                ("llama.context_length", GgufValue::Uint32(32_768)),
                ("llama.rope.scaling.type", GgufValue::Uint32(1)),
                ("llama.expert_count", GgufValue::Uint32(8)),
                ("llama.expert_used_count", GgufValue::Uint32(2)),
                ("mixtral.sliding_window", GgufValue::Uint32(4_096)),
            ],
        )
        .await;

        let arch = ArchitectureRegistry::create_model(model).unwrap();
        assert_eq!(arch.arch(), LlmArch::Mixtral);
        assert_eq!(arch.arch_name(), "mixtral");
        assert!(arch.features().contains(&"MoE"));
        assert!(arch.features().contains(&"Sliding Window"));
    }
}
