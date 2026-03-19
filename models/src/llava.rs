//! LLaVA multimodal adapter.

use crate::arch::LlmArch;
use crate::loader::Model;
use crate::vision::{ClipVisionEncoder, ImageInput, VisionEncoder};
use barq_core::error::{Error, Result};
use barq_core::tensor::{Shape, Tensor, TensorData, TensorType};
use std::sync::Arc;

/// LLaVA multimodal model wrapper.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use models::{loader::Model, LlavaModel};
///
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let model = Arc::new(Model::load("llava.gguf").await?);
/// let wrapper = LlavaModel::new(model)?;
/// let prompt = wrapper.insert_image_tokens("answer this", 2);
/// assert!(prompt.contains("<image>"));
/// # let _ = wrapper;
/// # Ok(())
/// # }
/// ```
pub struct LlavaModel {
    model: Arc<Model>,
    vision_encoder: ClipVisionEncoder,
}

impl LlavaModel {
    /// Create an LLaVA wrapper around a loaded model.
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Llava {
            return Err(Error::Unsupported(format!(
                "Expected LLaVA architecture, got {:?}",
                model.arch()
            )));
        }

        Ok(Self {
            model,
            vision_encoder: ClipVisionEncoder::new((224, 224), 16, 768),
        })
    }

    pub fn inner(&self) -> &Model {
        &self.model
    }

    pub fn vision_encoder(&self) -> &ClipVisionEncoder {
        &self.vision_encoder
    }

    /// Encode an image into patch embeddings.
    pub fn encode_image(&self, image: &ImageInput) -> Result<Tensor> {
        self.vision_encoder.encode(image)
    }

    /// Insert one or more `<image>` tokens into a text prompt.
    pub fn insert_image_tokens(&self, prompt: &str, images: usize) -> String {
        if images == 0 {
            return prompt.to_string();
        }

        let image_tokens = std::iter::repeat("<image>")
            .take(images)
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} {}", prompt.trim_end(), image_tokens)
    }

    /// Concatenate text and image embeddings along the sequence axis.
    pub fn fuse_text_and_image_embeddings(&self, text: &Tensor, image: &Tensor) -> Result<Tensor> {
        let text_dims = text.shape().dims();
        let image_dims = image.shape().dims();
        if text_dims.len() != 2 || image_dims.len() != 2 {
            return Err(Error::tensor("expected rank-2 embeddings"));
        }
        if text_dims[1] != image_dims[1] {
            return Err(Error::tensor("embedding dimensions must match"));
        }

        let mut combined =
            Vec::with_capacity(text.as_f32_slice()?.len() + image.as_f32_slice()?.len());
        combined.extend_from_slice(text.as_f32_slice()?);
        combined.extend_from_slice(image.as_f32_slice()?);

        Tensor::new(
            Some("llava_fused_embeddings".to_string()),
            TensorType::F32,
            Shape::new(vec![text_dims[0] + image_dims[0], text_dims[1]]),
            TensorData::F32(combined),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_test_gguf_file;
    use barq_core::gguf::GgufValue;

    fn llava_fixture() -> std::path::PathBuf {
        write_test_gguf_file(
            "llava",
            &[
                (
                    "general.architecture",
                    GgufValue::String("llava".to_string()),
                ),
                ("llama.block_count", GgufValue::Uint32(0)),
                ("llama.attention.head_count", GgufValue::Uint32(1)),
                ("llama.attention.head_count_kv", GgufValue::Uint32(1)),
                ("llama.embedding_length", GgufValue::Uint32(8)),
                ("llama.feed_forward_length", GgufValue::Uint32(16)),
                ("llama.context_length", GgufValue::Uint32(32)),
                ("llama.vocabulary_size", GgufValue::Uint32(128)),
                ("llama.rope.freq_base", GgufValue::Float32(10000.0)),
                ("llama.rope.freq_scale", GgufValue::Float32(1.0)),
                ("llama.rope.scaling.type", GgufValue::Uint32(0)),
            ],
        )
    }

    #[test]
    fn test_insert_image_tokens() {
        let model = Arc::new(
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(Model::load(llava_fixture()))
                .unwrap(),
        );
        let wrapper = LlavaModel::new(model).unwrap();
        let prompt = wrapper.insert_image_tokens("answer this", 1);
        assert!(prompt.contains("<image>"));
    }

    #[test]
    fn test_encode_and_fuse() {
        let model = Arc::new(
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(Model::load(llava_fixture()))
                .unwrap(),
        );
        let wrapper = LlavaModel::new(model).unwrap();
        let image = ImageInput::solid_rgb(32, 32, [0, 255, 0]);
        let encoded = wrapper.encode_image(&image).unwrap();
        assert_eq!(encoded.shape().dims()[1], 768);

        let fused = wrapper
            .fuse_text_and_image_embeddings(&encoded, &encoded)
            .unwrap();
        assert_eq!(fused.shape().dims()[0], encoded.shape().dims()[0] * 2);
    }
}
