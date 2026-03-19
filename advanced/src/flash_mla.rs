//! FlashMLA helpers for DeepSeek-style MLA models.
//!
//! This module does not implement a GPU kernel. Instead it encapsulates the
//! runtime decisions around MLA-2 cache-only and MLA-3 optimized execution:
//! - detect whether the loaded model actually uses MLA metadata
//! - estimate the memory savings from compressed latent attention caches
//! - derive an effective context window for `ContextParams`
//! - provide a CLI-friendly summary for users enabling `--mla`

use barq_core::error::Result;
use models::arch::LlmArch;
use models::context::ContextParams;
use models::loader::Model;
use serde::{Deserialize, Serialize};

/// FlashMLA operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlashMlaMode {
    /// MLA-2: cache-only path.
    Mla2CacheOnly,
    /// MLA-3: more aggressive compressed latent attention path.
    Mla3Optimized,
}

impl FlashMlaMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Mla2CacheOnly => "MLA-2 (cache-only)",
            Self::Mla3Optimized => "MLA-3 (optimized)",
        }
    }
}

/// Runtime report for FlashMLA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashMlaRuntime {
    pub arch: LlmArch,
    pub mode: FlashMlaMode,
    pub compression_ratio: f32,
    pub latent_heads: u32,
    pub n_layers: u32,
    pub head_dim: u32,
    pub recommended_ctx: usize,
}

impl FlashMlaRuntime {
    /// Build a runtime report from a loaded model.
    pub fn from_model(model: &Model, mode: FlashMlaMode) -> Option<Self> {
        match model.arch() {
            LlmArch::DeepSeek | LlmArch::DeepSeekMoE => {
                let base_ratio = model
                    .get_metadata("deepseek.mla_compression")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.25)
                    .clamp(0.0625, 1.0);
                let compression_ratio = match mode {
                    FlashMlaMode::Mla2CacheOnly => base_ratio,
                    FlashMlaMode::Mla3Optimized => (base_ratio * 0.75).clamp(0.0625, 1.0),
                };

                let latent_heads = model
                    .get_metadata("deepseek.n_latent_heads")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(model.hparams.n_head_kv);
                let head_dim = if model.hparams.n_head == 0 {
                    0
                } else {
                    model.hparams.n_embd / model.hparams.n_head
                };
                let recommended_ctx = Self::recommended_context(
                    model.hparams.n_ctx_train as usize,
                    compression_ratio,
                );

                Some(Self {
                    arch: model.arch(),
                    mode,
                    compression_ratio,
                    latent_heads,
                    n_layers: model.hparams.n_layer,
                    head_dim,
                    recommended_ctx,
                })
            }
            _ => None,
        }
    }

    /// Compute the effective context window for a given base context.
    pub fn recommended_context(base_ctx: usize, compression_ratio: f32) -> usize {
        if compression_ratio <= 0.0 {
            return base_ctx;
        }

        let boosted = (base_ctx as f32 / compression_ratio).ceil() as usize;
        boosted.max(base_ctx)
    }

    /// Estimate the compressed KV cache bytes for `n_tokens` tokens.
    pub fn estimated_kv_cache_bytes(&self, n_tokens: usize) -> usize {
        let bytes_per_value = 2usize; // use an FP16 baseline for estimation
        let baseline = n_tokens
            .saturating_mul(self.n_layers as usize)
            .saturating_mul(self.latent_heads as usize)
            .saturating_mul(self.head_dim as usize)
            .saturating_mul(2) // K + V
            .saturating_mul(bytes_per_value);
        (baseline as f32 * self.compression_ratio).round() as usize
    }

    /// Apply MLA-aware adjustments to a base context configuration.
    pub fn apply_to_context(&self, mut params: ContextParams) -> ContextParams {
        params.n_ctx =
            Self::recommended_context(params.n_ctx as usize, self.compression_ratio) as u32;
        params.flash_attn = true;
        params
    }

    /// Return a human-readable summary.
    pub fn summary_lines(&self) -> Vec<String> {
        vec![
            format!("Mode: {}", self.mode.label()),
            format!("Architecture: {:?}", self.arch),
            format!("Compression ratio: {:.3}", self.compression_ratio),
            format!("Latent heads: {}", self.latent_heads),
            format!("Head dim: {}", self.head_dim),
            format!("Recommended ctx: {}", self.recommended_ctx),
        ]
    }

    /// Print a compact summary to stdout.
    pub fn print_summary(&self) {
        println!("FlashMLA runtime:");
        for line in self.summary_lines() {
            println!("  {}", line);
        }
    }
}

/// Convenience helper for the CLI.
pub fn configure_context_for_mla(
    model: &Model,
    params: ContextParams,
    mode: FlashMlaMode,
) -> Result<Option<ContextParams>> {
    Ok(FlashMlaRuntime::from_model(model, mode).map(|runtime| runtime.apply_to_context(params)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use barq_core::gguf::GgufValue;
    use models::loader::Model;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_fixture(prefix: &str, kv_pairs: &[(&str, GgufValue)]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "barq-flashmla-{}-{}-{}",
            prefix,
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(barq_core::gguf::GGUF_MAGIC);
        bytes.extend_from_slice(&barq_core::gguf::GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());

        for (key, value) in kv_pairs {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&(value.get_type() as u32).to_le_bytes());
            match value {
                GgufValue::Uint32(v) => bytes.extend_from_slice(&v.to_le_bytes()),
                GgufValue::Float32(v) => bytes.extend_from_slice(&v.to_le_bytes()),
                GgufValue::String(v) => {
                    bytes.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    bytes.extend_from_slice(v.as_bytes());
                }
                _ => panic!("unsupported fixture value"),
            }
        }

        let alignment = barq_core::gguf::DEFAULT_ALIGNMENT as usize;
        let padding = (alignment - (bytes.len() % alignment)) % alignment;
        bytes.extend(std::iter::repeat_n(0u8, padding));

        let mut file = File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        path
    }

    async fn load_model(arch: &str) -> Model {
        let path = write_fixture(
            arch,
            &[
                ("general.architecture", GgufValue::String(arch.to_string())),
                ("deepseek.block_count", GgufValue::Uint32(28)),
                ("deepseek.attention.head_count", GgufValue::Uint32(32)),
                ("deepseek.attention.head_count_kv", GgufValue::Uint32(8)),
                ("deepseek.embedding_length", GgufValue::Uint32(4096)),
                ("deepseek.intermediate_size", GgufValue::Uint32(11_008)),
                ("deepseek.context_length", GgufValue::Uint32(131_072)),
                ("deepseek.rope.scaling.type", GgufValue::Uint32(2)),
                ("deepseek.mla_compression", GgufValue::Float32(0.25)),
                ("deepseek.n_latent_heads", GgufValue::Uint32(8)),
            ],
        );

        Model::load(&path).await.unwrap()
    }

    #[tokio::test]
    async fn test_flash_mla_runtime_from_model() {
        let model = load_model("deepseek").await;
        let runtime = FlashMlaRuntime::from_model(&model, FlashMlaMode::Mla2CacheOnly).unwrap();

        assert_eq!(runtime.arch, LlmArch::DeepSeek);
        assert!(runtime.compression_ratio <= 0.25);
        assert!(runtime.recommended_ctx >= model.hparams.n_ctx_train as usize);
        assert!(runtime.estimated_kv_cache_bytes(1024) > 0);
    }

    #[tokio::test]
    async fn test_flash_mla_apply_to_context() {
        let model = load_model("deepseek.moe").await;
        let runtime = FlashMlaRuntime::from_model(&model, FlashMlaMode::Mla3Optimized).unwrap();
        let params = ContextParams {
            n_ctx: 2048,
            ..Default::default()
        };

        let adjusted = runtime.apply_to_context(params);
        assert!(adjusted.n_ctx >= 2048);
        assert!(adjusted.flash_attn);
    }

    #[test]
    fn test_recommended_context_scales_up() {
        assert_eq!(FlashMlaRuntime::recommended_context(1024, 0.25), 4096);
        assert_eq!(FlashMlaRuntime::recommended_context(2048, 0.125), 16_384);
    }
}
