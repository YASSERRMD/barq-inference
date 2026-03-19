//! Model loader implementation
//!
//! Handles loading models from GGUF files and creating inference contexts.

use std::path::Path;
use std::sync::Arc;

use tokio::fs::File;

use crate::arch::LlmArch;
use barq_core::error::{Error, Result};
use barq_core::gguf::GgufReader;
use barq_core::tensor::Tensor;

/// Model hyperparameters
#[derive(Debug, Clone)]
pub struct ModelHParams {
    /// Number of layers
    pub n_layer: u32,
    /// Number of attention heads
    pub n_head: u32,
    /// Number of key-value heads
    pub n_head_kv: u32,
    /// Embedding dimension
    pub n_embd: u32,
    /// Intermediate/feed-forward dimension
    pub n_ff: u32,
    /// Vocabulary size
    pub n_vocab: u32,
    /// Maximum context length
    pub n_ctx_train: u32,
    /// Rotary embedding base frequency
    pub rope_freq_base: f32,
    /// Rotary embedding frequency scale
    pub rope_freq_scale: f32,
    /// Rope scaling type
    pub rope_scaling_type: u32,
}

impl Default for ModelHParams {
    fn default() -> Self {
        Self {
            n_layer: 32,
            n_head: 32,
            n_head_kv: 32,
            n_embd: 4096,
            n_ff: 11008,
            n_vocab: 32000,
            n_ctx_train: 2048,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            rope_scaling_type: 0,
        }
    }
}

/// Model loaded from GGUF file
#[derive(Clone)]
pub struct Model {
    /// Model architecture
    pub arch: LlmArch,
    /// Hyperparameters
    pub hparams: ModelHParams,
    /// Tensors by name
    tensors: Arc<tokio::sync::RwLock<std::collections::HashMap<String, Tensor>>>,
    /// GGUF metadata
    metadata: Arc<std::collections::HashMap<String, String>>,
}

impl Model {
    /// Load a model from a GGUF file
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let _file = File::open(&path_str).await.map_err(Error::Io)?;
        // For now, we'll use synchronous GGUF reading
        // In production, this should be fully async

        // Use blocking task for file reading and tensor loading
        let (arch, hparams, tensors, metadata) = tokio::task::spawn_blocking(move || {
            // Open the GGUF file
            let mut reader = GgufReader::open(&path_str)?;

            // Determine architecture from metadata
            let arch = Self::detect_arch(&reader);

            // Extract hyperparameters
            let hparams = Self::extract_hparams(&reader);

            // Extract metadata
            let mut metadata_map = std::collections::HashMap::new();
            for (key, value) in reader.kv_pairs() {
                use barq_core::gguf::GgufValue;
                match value {
                    GgufValue::String(s) => {
                        metadata_map.insert(key.clone(), s.clone());
                    }
                    GgufValue::Uint32(u) => {
                        metadata_map.insert(key.clone(), u.to_string());
                    }
                    GgufValue::Int32(i) => {
                        metadata_map.insert(key.clone(), i.to_string());
                    }
                    GgufValue::Float32(f) => {
                        metadata_map.insert(key.clone(), f.to_string());
                    }
                    GgufValue::Bool(b) => {
                        metadata_map.insert(key.clone(), b.to_string());
                    }
                    // Handle arrays - especially tokenizer tokens
                    GgufValue::Array(arr) => {
                        if key == "tokenizer.ggml.tokens" {
                            // Convert token array to JSON for storage
                            let tokens: Vec<String> = arr
                                .iter()
                                .filter_map(|v| {
                                    if let GgufValue::String(s) = v {
                                        Some(s.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if let Ok(json) = serde_json::to_string(&tokens) {
                                metadata_map.insert(key.clone(), json);
                            }
                        } else {
                            // For other arrays, store count
                            metadata_map
                                .insert(key.clone(), format!("[array; {} items]", arr.len()));
                        }
                    }
                    _ => {
                        // Store other types as string representation
                        metadata_map.insert(key.clone(), format!("{:?}", value));
                    }
                }
            }

            // Load all tensors
            let mut tensors_map = std::collections::HashMap::new();
            let tensor_infos = reader.tensor_info().to_vec();

            for info in &tensor_infos {
                match reader.load_tensor(&info.name) {
                    Ok(tensor) => {
                        tensors_map.insert(info.name.clone(), tensor);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load tensor {}: {}", info.name, e);
                    }
                }
            }

            Ok::<
                (
                    LlmArch,
                    ModelHParams,
                    std::collections::HashMap<String, Tensor>,
                    std::collections::HashMap<String, String>,
                ),
                Error,
            >((arch, hparams, tensors_map, metadata_map))
        })
        .await
        .map_err(|e| Error::Backend(format!("Failed to join task: {}", e)))??;

        println!("Loaded {} tensors from GGUF file", tensors.len());

        Ok(Self {
            arch,
            hparams,
            tensors: Arc::new(tokio::sync::RwLock::new(tensors)),
            metadata: Arc::new(metadata),
        })
    }

    fn detect_arch(reader: &GgufReader) -> LlmArch {
        use barq_core::gguf::GgufValue;
        if let Some(GgufValue::String(arch_name)) = reader.get("general.architecture") {
            match arch_name.to_lowercase().as_str() {
                "llama" => LlmArch::Llama,
                "mistral" => LlmArch::Mistral,
                "mixtral" => LlmArch::Mixtral,
                "gpt2" => LlmArch::Gpt2,
                "gptneox" => LlmArch::GptNeoX,
                "bert" => LlmArch::Bert,
                "t5" => LlmArch::T5,
                "bloom" => LlmArch::Bloom,
                "falcon" => LlmArch::Falcon,
                "mpt" => LlmArch::Mpt,
                "phi" => LlmArch::Phi,
                "phi2" => LlmArch::Phi2,
                "phi3" => LlmArch::Phi3,
                "qwen" => LlmArch::Qwen,
                "qwen2" => LlmArch::Qwen2,
                "qwen2.moe" => LlmArch::Qwen2Moe,
                "qwen3" => LlmArch::Qwen3,
                "deepseek" => LlmArch::DeepSeek,
                "deepseek.moe" => LlmArch::DeepSeekMoE,
                "gemma" => LlmArch::Gemma,
                "gemma2" => LlmArch::Gemma2,
                _ => LlmArch::Unknown,
            }
        } else {
            LlmArch::Unknown
        }
    }

    fn extract_hparams(reader: &GgufReader) -> ModelHParams {
        let get_u32 = |key: &str| -> u32 {
            reader
                .get(key)
                .and_then(|v| {
                    if let barq_core::gguf::GgufValue::Uint32(u) = v {
                        Some(*u)
                    } else {
                        None
                    }
                })
                .unwrap_or(0)
        };

        let get_f32 = |key: &str| -> f32 {
            reader
                .get(key)
                .and_then(|v| {
                    if let barq_core::gguf::GgufValue::Float32(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0)
        };

        // Detect architecture to use correct keys
        let arch_name = reader
            .get("general.architecture")
            .and_then(|v| {
                if let barq_core::gguf::GgufValue::String(s) = v {
                    Some(s.to_lowercase())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let is_qwen = arch_name.contains("qwen");
        let is_deepseek = arch_name.contains("deepseek");

        // Try both architecture-specific and general prefixes for vocab size
        let n_vocab = get_u32("llama.vocabulary_size");
        let n_vocab = if n_vocab == 0 {
            let qwen_vocab = get_u32("qwen.vocabulary_size");
            let qwen_vocab = if qwen_vocab == 0 && is_deepseek {
                get_u32("deepseek.vocabulary_size")
            } else {
                qwen_vocab
            };
            if qwen_vocab == 0 {
                reader
                    .get("general.vocab_size")
                    .and_then(|v| {
                        if let barq_core::gguf::GgufValue::Uint32(u) = v {
                            Some(*u)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(102400) // DeepSeek uses larger vocab
            } else {
                qwen_vocab
            }
        } else {
            n_vocab
        };

        ModelHParams {
            n_layer: {
                let v = get_u32("llama.block_count");
                let v = if v == 0 && is_qwen {
                    get_u32("qwen.block_count")
                } else {
                    v
                };
                let v = if v == 0 && is_deepseek {
                    get_u32("deepseek.block_count")
                } else {
                    v
                };
                if v == 0 {
                    get_u32("general.n_layers")
                } else {
                    v
                }
            },
            n_head: {
                let v = get_u32("llama.attention.head_count");
                let v = if v == 0 && is_qwen {
                    get_u32("qwen.attention.head_count")
                } else {
                    v
                };
                let v = if v == 0 && is_deepseek {
                    get_u32("deepseek.attention.head_count")
                } else {
                    v
                };
                if v == 0 {
                    get_u32("general.n_head")
                } else {
                    v
                }
            },
            n_head_kv: {
                let v = get_u32("llama.attention.head_count_kv");
                let v = if v == 0 && is_qwen {
                    get_u32("qwen.attention.head_count_kv")
                } else {
                    v
                };
                let v = if v == 0 && is_deepseek {
                    get_u32("deepseek.attention.head_count_kv")
                } else {
                    v
                };
                // If still 0, default to n_head (full attention)
                if v == 0 {
                    get_u32("llama.attention.head_count")
                } else {
                    v
                }
            },
            n_embd: {
                let v = get_u32("llama.embedding_length");
                let v = if v == 0 && is_qwen {
                    get_u32("qwen.embedding_length")
                } else {
                    v
                };
                let v = if v == 0 && is_deepseek {
                    get_u32("deepseek.embedding_length")
                } else {
                    v
                };
                if v == 0 {
                    get_u32("general.n_embd")
                } else {
                    v
                }
            },
            n_ff: {
                let v = get_u32("llama.feed_forward_length");
                let v = if v == 0 && is_qwen {
                    get_u32("qwen.intermediate_size")
                } else {
                    v
                };
                let v = if v == 0 && is_deepseek {
                    get_u32("deepseek.intermediate_size")
                } else {
                    v
                };
                if v == 0 {
                    get_u32("general.n_ff")
                } else {
                    v
                }
            },
            n_vocab,
            n_ctx_train: {
                let v = get_u32("llama.context_length");
                let v = if v == 0 && is_qwen {
                    get_u32("qwen.context_length")
                } else {
                    v
                };
                let v = if v == 0 && is_deepseek {
                    get_u32("deepseek.context_length")
                } else {
                    v
                };
                if v == 0 {
                    get_u32("general.n_context")
                } else {
                    v
                }
            },
            rope_freq_base: {
                let v = get_f32("llama.rope.freq_base");
                let v = if v == 0.0 && is_qwen {
                    get_f32("qwen.rope.freq_base")
                } else {
                    v
                };
                let v = if v == 0.0 && is_deepseek {
                    get_f32("deepseek.rope.freq_base")
                } else {
                    v
                };
                // Qwen and DeepSeek use different RoPE bases
                if v == 0.0 {
                    if is_qwen {
                        1000000.0
                    } else if is_deepseek {
                        10000.0 // DeepSeek uses standard base with Yarn scaling
                    } else {
                        10000.0
                    }
                } else {
                    v
                }
            },
            rope_freq_scale: {
                let v = get_f32("llama.rope.freq_scale");
                let v = if v == 0.0 && is_qwen {
                    get_f32("qwen.rope.freq_scale")
                } else {
                    v
                };
                let v = if v == 0.0 && is_deepseek {
                    get_f32("deepseek.rope.freq_scale")
                } else {
                    v
                };
                if v == 0.0 {
                    1.0
                } else {
                    v
                }
            },
            rope_scaling_type: {
                let v = get_u32("llama.rope.scaling.type");
                let v = if v == 0 && is_qwen {
                    get_u32("qwen.rope.scaling.type")
                } else {
                    v
                };
                let v = if v == 0 && is_deepseek {
                    get_u32("deepseek.rope.scaling.type")
                } else {
                    v
                };
                v
            },
            ..Default::default()
        }
    }

    /// Get a tensor by name
    pub async fn get_tensor(&self, name: &str) -> Option<Tensor> {
        let tensors = self.tensors.read().await;
        tensors.get(name).cloned()
    }

    /// Returns the model architecture
    pub fn arch(&self) -> LlmArch {
        self.arch
    }

    /// Returns the hyperparameters
    pub fn hparams(&self) -> &ModelHParams {
        &self.hparams
    }

    /// Returns metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Get a tensor by name (synchronous, blocking)
    pub fn get_tensor_blocking(&self, name: &str) -> Option<Tensor> {
        let tensors = self.tensors.try_read().ok()?;
        tensors.get(name).cloned()
    }

    /// Returns all metadata
    pub fn metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.metadata
    }
}

/// Model loader trait
#[async_trait::async_trait]
pub trait ModelLoader: Send + Sync {
    /// Load a model from path
    async fn load<P: AsRef<Path> + Send>(&self, path: P) -> Result<Model>;

    /// Load a model with progress callback
    async fn load_with_progress<P: AsRef<Path> + Send>(
        &self,
        path: P,
        progress: Box<dyn Fn(f32) + Send + Sync>,
    ) -> Result<Model>;
}

/// Default model loader
pub struct DefaultModelLoader;

#[async_trait::async_trait]
impl ModelLoader for DefaultModelLoader {
    async fn load<P: AsRef<Path> + Send>(&self, path: P) -> Result<Model> {
        Model::load(path).await
    }

    async fn load_with_progress<P: AsRef<Path> + Send>(
        &self,
        path: P,
        progress: Box<dyn Fn(f32) + Send + Sync>,
    ) -> Result<Model> {
        progress(0.0);
        let model = Model::load(path).await?;
        progress(1.0);
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::LlmArch;
    use crate::test_support::write_test_gguf_file;
    use barq_core::gguf::GgufValue;

    #[tokio::test]
    async fn test_hparams_defaults() {
        let hparams = ModelHParams::default();
        assert_eq!(hparams.n_layer, 32);
        assert_eq!(hparams.n_embd, 4096);
    }

    #[tokio::test]
    async fn test_load_qwen_family_fixtures() {
        let qwen_path = write_test_gguf_file(
            "qwen",
            &[
                (
                    "general.architecture",
                    GgufValue::String("qwen".to_string()),
                ),
                ("llama.block_count", GgufValue::Uint32(24)),
                ("llama.attention.head_count", GgufValue::Uint32(32)),
                ("llama.attention.head_count_kv", GgufValue::Uint32(32)),
                ("llama.embedding_length", GgufValue::Uint32(4096)),
                ("llama.feed_forward_length", GgufValue::Uint32(11008)),
                ("llama.context_length", GgufValue::Uint32(32768)),
                ("qwen.rope.freq_base", GgufValue::Float32(1_000_000.0)),
                ("qwen.rope.scaling.type", GgufValue::Uint32(1)),
            ],
        );

        let qwen = Model::load(&qwen_path).await.unwrap();
        assert_eq!(qwen.arch(), LlmArch::Qwen);
        assert_eq!(qwen.hparams.n_layer, 24);
        assert_eq!(qwen.hparams.n_head, 32);
        assert_eq!(qwen.hparams.n_head_kv, 32);
        assert_eq!(qwen.hparams.n_embd, 4096);
        assert_eq!(qwen.hparams.n_ff, 11008);
        assert_eq!(qwen.hparams.n_ctx_train, 32768);
        assert_eq!(qwen.hparams.rope_freq_base, 1_000_000.0);
        assert_eq!(qwen.hparams.rope_scaling_type, 1);

        let qwen2_path = write_test_gguf_file(
            "qwen2",
            &[
                (
                    "general.architecture",
                    GgufValue::String("qwen2".to_string()),
                ),
                ("general.vocab_size", GgufValue::Uint32(151_936)),
                ("qwen.block_count", GgufValue::Uint32(28)),
                ("qwen.attention.head_count", GgufValue::Uint32(40)),
                ("qwen.attention.head_count_kv", GgufValue::Uint32(8)),
                ("qwen.embedding_length", GgufValue::Uint32(5120)),
                ("qwen.intermediate_size", GgufValue::Uint32(13_824)),
                ("qwen.context_length", GgufValue::Uint32(32_768)),
                ("qwen.rope.freq_base", GgufValue::Float32(1_000_000.0)),
                ("qwen.rope.scaling.type", GgufValue::Uint32(1)),
            ],
        );

        let qwen2 = Model::load(&qwen2_path).await.unwrap();
        assert_eq!(qwen2.arch(), LlmArch::Qwen2);
        assert_eq!(qwen2.hparams.n_layer, 28);
        assert_eq!(qwen2.hparams.n_head, 40);
        assert_eq!(qwen2.hparams.n_head_kv, 8);
        assert_eq!(qwen2.hparams.n_embd, 5120);
        assert_eq!(qwen2.hparams.n_ff, 13_824);
        assert_eq!(qwen2.hparams.n_vocab, 151_936);
        assert_eq!(qwen2.hparams.n_ctx_train, 32_768);
        assert_eq!(qwen2.hparams.rope_scaling_type, 1);

        let qwen3_path = write_test_gguf_file(
            "qwen3",
            &[
                (
                    "general.architecture",
                    GgufValue::String("qwen3".to_string()),
                ),
                ("qwen.block_count", GgufValue::Uint32(32)),
                ("qwen.attention.head_count", GgufValue::Uint32(64)),
                ("qwen.attention.head_count_kv", GgufValue::Uint32(8)),
                ("qwen.embedding_length", GgufValue::Uint32(8192)),
                ("qwen.intermediate_size", GgufValue::Uint32(22_016)),
                ("qwen.context_length", GgufValue::Uint32(131_072)),
                ("qwen.rope.freq_base", GgufValue::Float32(1_000_000.0)),
                ("qwen.rope.scaling.type", GgufValue::Uint32(1)),
            ],
        );

        let qwen3 = Model::load(&qwen3_path).await.unwrap();
        assert_eq!(qwen3.arch(), LlmArch::Qwen3);
        assert_eq!(qwen3.hparams.n_layer, 32);
        assert_eq!(qwen3.hparams.n_head, 64);
        assert_eq!(qwen3.hparams.n_head_kv, 8);
        assert_eq!(qwen3.hparams.n_embd, 8192);
        assert_eq!(qwen3.hparams.n_ff, 22_016);
        assert_eq!(qwen3.hparams.n_ctx_train, 131_072);
        assert_eq!(qwen3.hparams.rope_scaling_type, 1);
    }
}
