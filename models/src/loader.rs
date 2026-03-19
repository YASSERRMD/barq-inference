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

/// Model loaded from a GGUF file.
#[derive(Clone)]
pub struct Model {
    /// Model architecture
    pub arch: LlmArch,
    /// Hyperparameters
    pub hparams: ModelHParams,
    /// Total number of tensors loaded from the GGUF file
    tensor_count: usize,
    /// Tensors by name
    tensors: Arc<tokio::sync::RwLock<std::collections::HashMap<String, Tensor>>>,
    /// GGUF metadata
    metadata: Arc<std::collections::HashMap<String, String>>,
}

impl Model {
    /// Load a model from a GGUF file.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use models::loader::Model;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let model = Model::load("path/to/model.gguf").await?;
    ///     println!("loaded {:?}", model.arch());
    ///     Ok(())
    /// }
    /// ```
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let _file = File::open(&path_str).await.map_err(Error::Io)?;
        // For now, we'll use synchronous GGUF reading
        // In production, this should be fully async

        // Use blocking task for file reading and tensor loading
        let (arch, hparams, tensor_count, tensors, metadata) =
            tokio::task::spawn_blocking(move || {
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
                        // Handle arrays - especially tokenizer tokens, merges, and token types.
                        GgufValue::Array(arr) => {
                            if key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges" {
                                let items: Vec<String> = arr
                                    .iter()
                                    .filter_map(|v| {
                                        if let GgufValue::String(s) = v {
                                            Some(s.clone())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                if let Ok(json) = serde_json::to_string(&items) {
                                    metadata_map.insert(key.clone(), json);
                                }
                            } else if key == "tokenizer.ggml.token_type" {
                                let token_types: Vec<i32> = arr
                                    .iter()
                                    .filter_map(|v| match v {
                                        GgufValue::Int32(i) => Some(*i),
                                        GgufValue::Uint32(u) => Some(*u as i32),
                                        _ => None,
                                    })
                                    .collect();
                                if let Ok(json) = serde_json::to_string(&token_types) {
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
                        usize,
                        std::collections::HashMap<String, Tensor>,
                        std::collections::HashMap<String, String>,
                    ),
                    Error,
                >((arch, hparams, tensors_map.len(), tensors_map, metadata_map))
            })
            .await
            .map_err(|e| Error::Backend(format!("Failed to join task: {}", e)))??;

        println!("Loaded {} tensors from GGUF file", tensor_count);

        Ok(Self {
            arch,
            hparams,
            tensor_count,
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
                "qwen2vl" | "qwen2.vl" | "qwen-vl" => LlmArch::Qwen2Vl,
                "llava" | "llava-1.5" | "llava-1.6" => LlmArch::Llava,
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

        let get_u32_any = |keys: &[&str]| -> u32 {
            keys.iter()
                .find_map(|key| {
                    let value = get_u32(key);
                    if value == 0 {
                        None
                    } else {
                        Some(value)
                    }
                })
                .unwrap_or(0)
        };

        let get_f32_any = |keys: &[&str]| -> f32 {
            keys.iter()
                .find_map(|key| {
                    let value = get_f32(key);
                    if value == 0.0 {
                        None
                    } else {
                        Some(value)
                    }
                })
                .unwrap_or(0.0)
        };

        let is_deepseek = arch_name.contains("deepseek");

        let tokenizer_vocab = reader
            .get("tokenizer.ggml.tokens")
            .and_then(|value| match value {
                barq_core::gguf::GgufValue::Array(tokens) => Some(tokens.len() as u32),
                _ => None,
            })
            .unwrap_or(0);

        // Try both architecture-specific and general prefixes for vocab size.
        let n_vocab = get_u32_any(&[
            "llama.vocabulary_size",
            "qwen.vocabulary_size",
            "qwen2.vocabulary_size",
            "qwen3.vocabulary_size",
            "deepseek.vocabulary_size",
            "general.vocab_size",
        ]);
        let n_vocab = if n_vocab == 0 {
            if tokenizer_vocab > 0 {
                tokenizer_vocab
            } else if is_deepseek {
                102400 // DeepSeek uses larger vocab
            } else {
                32000
            }
        } else {
            n_vocab
        };

        ModelHParams {
            n_layer: get_u32_any(&[
                "llama.block_count",
                "qwen.block_count",
                "qwen2.block_count",
                "qwen3.block_count",
                "deepseek.block_count",
                "general.n_layers",
            ]),
            n_head: get_u32_any(&[
                "llama.attention.head_count",
                "qwen.attention.head_count",
                "qwen2.attention.head_count",
                "qwen3.attention.head_count",
                "deepseek.attention.head_count",
                "general.n_head",
            ]),
            n_head_kv: {
                let v = get_u32_any(&[
                    "llama.attention.head_count_kv",
                    "qwen.attention.head_count_kv",
                    "qwen2.attention.head_count_kv",
                    "qwen3.attention.head_count_kv",
                    "deepseek.attention.head_count_kv",
                ]);
                // If still 0, default to full attention.
                if v == 0 {
                    get_u32_any(&[
                        "llama.attention.head_count",
                        "qwen.attention.head_count",
                        "qwen2.attention.head_count",
                        "qwen3.attention.head_count",
                        "deepseek.attention.head_count",
                    ])
                } else {
                    v
                }
            },
            n_embd: get_u32_any(&[
                "llama.embedding_length",
                "qwen.embedding_length",
                "qwen2.embedding_length",
                "qwen3.embedding_length",
                "deepseek.embedding_length",
                "general.n_embd",
            ]),
            n_ff: get_u32_any(&[
                "llama.feed_forward_length",
                "qwen.intermediate_size",
                "qwen2.feed_forward_length",
                "qwen2.intermediate_size",
                "qwen3.intermediate_size",
                "deepseek.intermediate_size",
                "general.n_ff",
            ]),
            n_vocab,
            n_ctx_train: get_u32_any(&[
                "llama.context_length",
                "qwen.context_length",
                "qwen2.context_length",
                "qwen3.context_length",
                "deepseek.context_length",
                "general.n_context",
            ]),
            rope_freq_base: {
                let v = get_f32_any(&[
                    "llama.rope.freq_base",
                    "qwen.rope.freq_base",
                    "qwen2.rope.freq_base",
                    "qwen3.rope.freq_base",
                    "deepseek.rope.freq_base",
                ]);
                // Qwen and DeepSeek use different RoPE bases
                if v == 0.0 {
                    if arch_name.contains("qwen") {
                        1_000_000.0
                    } else if is_deepseek {
                        10_000.0 // DeepSeek uses standard base with Yarn scaling
                    } else {
                        10_000.0
                    }
                } else {
                    v
                }
            },
            rope_freq_scale: {
                let v = get_f32_any(&[
                    "llama.rope.freq_scale",
                    "qwen.rope.freq_scale",
                    "qwen2.rope.freq_scale",
                    "qwen3.rope.freq_scale",
                    "deepseek.rope.freq_scale",
                ]);
                if v == 0.0 {
                    1.0
                } else {
                    v
                }
            },
            rope_scaling_type: get_u32_any(&[
                "llama.rope.scaling.type",
                "qwen.rope.scaling.type",
                "qwen2.rope.scaling.type",
                "qwen3.rope.scaling.type",
                "deepseek.rope.scaling.type",
            ]),
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

    /// Returns the number of tensors loaded into the model
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
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
                ("qwen2.block_count", GgufValue::Uint32(28)),
                ("qwen2.attention.head_count", GgufValue::Uint32(40)),
                ("qwen2.attention.head_count_kv", GgufValue::Uint32(8)),
                ("qwen2.embedding_length", GgufValue::Uint32(5120)),
                ("qwen2.feed_forward_length", GgufValue::Uint32(13_824)),
                ("qwen2.context_length", GgufValue::Uint32(32_768)),
                ("qwen2.rope.freq_base", GgufValue::Float32(1_000_000.0)),
                ("qwen2.rope.scaling.type", GgufValue::Uint32(1)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![
                        GgufValue::String("<unk>".to_string()),
                        GgufValue::String("hello".to_string()),
                        GgufValue::String("world".to_string()),
                        GgufValue::String("</s>".to_string()),
                    ]),
                ),
                (
                    "tokenizer.ggml.merges",
                    GgufValue::Array(vec![
                        GgufValue::String("h e".to_string()),
                        GgufValue::String("he l".to_string()),
                    ]),
                ),
                (
                    "tokenizer.ggml.token_type",
                    GgufValue::Array(vec![
                        GgufValue::Int32(1),
                        GgufValue::Int32(3),
                        GgufValue::Int32(3),
                        GgufValue::Int32(1),
                    ]),
                ),
            ],
        );

        let qwen2 = Model::load(&qwen2_path).await.unwrap();
        assert_eq!(qwen2.arch(), LlmArch::Qwen2);
        assert_eq!(qwen2.hparams.n_layer, 28);
        assert_eq!(qwen2.hparams.n_head, 40);
        assert_eq!(qwen2.hparams.n_head_kv, 8);
        assert_eq!(qwen2.hparams.n_embd, 5120);
        assert_eq!(qwen2.hparams.n_ff, 13_824);
        assert_eq!(qwen2.hparams.n_vocab, 4);
        assert_eq!(qwen2.hparams.n_ctx_train, 32_768);
        assert_eq!(qwen2.hparams.rope_scaling_type, 1);
        let expected_tokens = serde_json::to_string(&vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
            "</s>".to_string(),
        ])
        .unwrap();
        let expected_merges =
            serde_json::to_string(&vec!["h e".to_string(), "he l".to_string()]).unwrap();
        let expected_token_types = serde_json::to_string(&vec![1_i32, 3, 3, 1]).unwrap();
        assert_eq!(
            qwen2
                .metadata()
                .get("tokenizer.ggml.tokens")
                .map(|s| s.as_str()),
            Some(expected_tokens.as_str())
        );
        assert_eq!(
            qwen2
                .metadata()
                .get("tokenizer.ggml.merges")
                .map(|s| s.as_str()),
            Some(expected_merges.as_str())
        );
        assert_eq!(
            qwen2
                .metadata()
                .get("tokenizer.ggml.token_type")
                .map(|s| s.as_str()),
            Some(expected_token_types.as_str())
        );

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
