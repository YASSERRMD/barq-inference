//! Qwen2-VL multimodal adapter.

use crate::arch::LlmArch;
use crate::loader::Model;
use crate::vision::{ClipVisionEncoder, ImageInput, VisionEncoder};
use barq_core::error::{Error, Result};
use barq_core::tensor::{Shape, Tensor, TensorData, TensorType};
use std::sync::Arc;

/// Qwen2-VL multimodal model wrapper.
pub struct Qwen2VlModel {
    model: Arc<Model>,
    vision_encoder: ClipVisionEncoder,
}

impl Qwen2VlModel {
    pub fn new(model: Arc<Model>) -> Result<Self> {
        if model.arch() != LlmArch::Qwen2Vl {
            return Err(Error::Unsupported(format!(
                "Expected Qwen2-VL architecture, got {:?}",
                model.arch()
            )));
        }

        Ok(Self {
            model,
            vision_encoder: ClipVisionEncoder::new((224, 224), 14, 1024),
        })
    }

    pub fn inner(&self) -> &Model {
        &self.model
    }

    pub fn vision_encoder(&self) -> &ClipVisionEncoder {
        &self.vision_encoder
    }

    pub fn encode_image(&self, image: &ImageInput) -> Result<Tensor> {
        self.vision_encoder.encode(image)
    }

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
            Some("qwen2vl_fused_embeddings".to_string()),
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

    fn qwen2vl_fixture() -> std::path::PathBuf {
        write_test_gguf_file(
            "qwen2vl",
            &[
                (
                    "general.architecture",
                    GgufValue::String("qwen2vl".to_string()),
                ),
                ("qwen.block_count", GgufValue::Uint32(0)),
                ("qwen.attention.head_count", GgufValue::Uint32(1)),
                ("qwen.attention.head_count_kv", GgufValue::Uint32(1)),
                ("qwen.embedding_length", GgufValue::Uint32(8)),
                ("qwen.intermediate_size", GgufValue::Uint32(16)),
                ("qwen.context_length", GgufValue::Uint32(32)),
                ("qwen.vocabulary_size", GgufValue::Uint32(128)),
                ("qwen.rope.freq_base", GgufValue::Float32(1_000_000.0)),
                ("qwen.rope.freq_scale", GgufValue::Float32(1.0)),
                ("qwen.rope.scaling.type", GgufValue::Uint32(1)),
            ],
        )
    }

    #[test]
    fn test_insert_image_tokens() {
        let model = Arc::new(
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(Model::load(qwen2vl_fixture()))
                .unwrap(),
        );
        let wrapper = Qwen2VlModel::new(model).unwrap();
        let prompt = wrapper.insert_image_tokens("describe this", 2);
        assert!(prompt.contains("<image>"));
    }

    #[test]
    fn test_encode_and_fuse() {
        let model = Arc::new(
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(Model::load(qwen2vl_fixture()))
                .unwrap(),
        );
        let wrapper = Qwen2VlModel::new(model).unwrap();
        let image = ImageInput::solid_rgb(32, 32, [255, 0, 0]);
        let encoded = wrapper.encode_image(&image).unwrap();
        assert_eq!(encoded.shape().dims()[1], 1024);

        let fused = wrapper
            .fuse_text_and_image_embeddings(&encoded, &encoded)
            .unwrap();
        assert_eq!(fused.shape().dims()[0], encoded.shape().dims()[0] * 2);
    }
}
