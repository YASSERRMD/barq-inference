//! Vision encoder support for multimodal models.
//!
//! This module provides a small, deterministic vision preprocessing and
//! embedding pipeline that can be used by future multimodal adapters.

use barq_core::error::{Error, Result};
use barq_core::tensor::{Shape, Tensor, TensorData, TensorType};

/// Raw RGB image input.
///
/// # Example
///
/// ```rust
/// use models::ImageInput;
///
/// let image = ImageInput::solid_rgb(2, 2, [255, 0, 0]);
/// assert_eq!(image.width, 2);
/// assert_eq!(image.pixels.len(), 12);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl ImageInput {
    /// Build an RGB image from raw bytes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use models::ImageInput;
    ///
    /// let image = ImageInput::new(1, 1, vec![0, 128, 255]).unwrap();
    /// assert_eq!(image.height, 1);
    /// ```
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|px| px.checked_mul(3))
            .ok_or_else(|| Error::tensor("image dimensions overflow"))?;
        if pixels.len() != expected {
            return Err(Error::tensor(format!(
                "expected {} RGB bytes, got {}",
                expected,
                pixels.len()
            )));
        }

        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Create a solid-color RGB image.
    ///
    /// # Example
    ///
    /// ```rust
    /// use models::ImageInput;
    ///
    /// let image = ImageInput::solid_rgb(2, 1, [1, 2, 3]);
    /// assert_eq!(image.pixels, vec![1, 2, 3, 1, 2, 3]);
    /// ```
    pub fn solid_rgb(width: usize, height: usize, rgb: [u8; 3]) -> Self {
        let mut pixels = Vec::with_capacity(width * height * 3);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&rgb);
        }

        Self {
            width,
            height,
            pixels,
        }
    }

    /// Resize the image with nearest-neighbor sampling.
    ///
    /// # Example
    ///
    /// ```rust
    /// use models::ImageInput;
    ///
    /// let image = ImageInput::solid_rgb(1, 1, [7, 8, 9]);
    /// let resized = image.resize_nearest(2, 2).unwrap();
    /// assert_eq!(resized.width, 2);
    /// assert_eq!(resized.height, 2);
    /// ```
    pub fn resize_nearest(&self, target_width: usize, target_height: usize) -> Result<Self> {
        if target_width == 0 || target_height == 0 {
            return Err(Error::tensor("image dimensions must be non-zero"));
        }

        if self.width == target_width && self.height == target_height {
            return Ok(self.clone());
        }

        let mut pixels = Vec::with_capacity(target_width * target_height * 3);
        for y in 0..target_height {
            let src_y = y * self.height / target_height;
            for x in 0..target_width {
                let src_x = x * self.width / target_width;
                let src_idx = (src_y * self.width + src_x) * 3;
                pixels.extend_from_slice(&self.pixels[src_idx..src_idx + 3]);
            }
        }

        Ok(Self {
            width: target_width,
            height: target_height,
            pixels,
        })
    }

    /// Convert the image bytes to normalized floating-point values.
    pub fn normalize(&self) -> Vec<f32> {
        self.pixels
            .iter()
            .map(|&value| value as f32 / 255.0)
            .collect()
    }
}

/// Shared preprocessing options for vision encoders.
///
/// # Example
///
/// ```rust
/// use models::{ImageInput, ImagePreprocessor};
///
/// let image = ImageInput::solid_rgb(2, 2, [255, 0, 0]);
/// let preprocessor = ImagePreprocessor::new(1, 1, true);
/// let pixels = preprocessor.preprocess(&image).unwrap();
/// assert_eq!(pixels.len(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagePreprocessor {
    pub target_width: usize,
    pub target_height: usize,
    pub normalize: bool,
}

impl ImagePreprocessor {
    /// Create a new image preprocessor.
    pub fn new(target_width: usize, target_height: usize, normalize: bool) -> Self {
        Self {
            target_width,
            target_height,
            normalize,
        }
    }

    /// Resize and optionally normalize an image into a flat float buffer.
    pub fn preprocess(&self, image: &ImageInput) -> Result<Vec<f32>> {
        let resized = image.resize_nearest(self.target_width, self.target_height)?;
        Ok(if self.normalize {
            resized.normalize()
        } else {
            resized.pixels.iter().map(|&value| value as f32).collect()
        })
    }
}

/// Vision encoder interface.
///
/// Implementations convert an [`ImageInput`] into patch embeddings.
pub trait VisionEncoder: Send + Sync {
    fn image_size(&self) -> (usize, usize);
    fn patch_size(&self) -> usize;
    fn embedding_dim(&self) -> usize;
    fn preprocess(&self, image: &ImageInput) -> Result<Vec<f32>>;
    fn encode(&self, image: &ImageInput) -> Result<Tensor>;
}

/// A deterministic CLIP-like vision encoder used for multimodal scaffolding.
///
/// # Example
///
/// ```rust
/// use models::vision::{ClipVisionEncoder, ImageInput, VisionEncoder};
///
/// let encoder = ClipVisionEncoder::new((224, 224), 16, 768);
/// let image = ImageInput::solid_rgb(224, 224, [0, 255, 0]);
/// let embeddings = encoder.encode(&image).unwrap();
/// assert_eq!(embeddings.shape().dims()[1], 768);
/// ```
#[derive(Debug, Clone)]
pub struct ClipVisionEncoder {
    preprocessor: ImagePreprocessor,
    patch_size: usize,
    embedding_dim: usize,
}

impl ClipVisionEncoder {
    /// Create a CLIP-like encoder with a fixed image size and patch layout.
    pub fn new(image_size: (usize, usize), patch_size: usize, embedding_dim: usize) -> Self {
        Self {
            preprocessor: ImagePreprocessor::new(image_size.0, image_size.1, true),
            patch_size,
            embedding_dim,
        }
    }

    fn patches_per_axis(&self) -> (usize, usize) {
        (
            self.preprocessor.target_width / self.patch_size,
            self.preprocessor.target_height / self.patch_size,
        )
    }
}

impl VisionEncoder for ClipVisionEncoder {
    fn image_size(&self) -> (usize, usize) {
        (
            self.preprocessor.target_width,
            self.preprocessor.target_height,
        )
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn preprocess(&self, image: &ImageInput) -> Result<Vec<f32>> {
        self.preprocessor.preprocess(image)
    }

    fn encode(&self, image: &ImageInput) -> Result<Tensor> {
        let preprocessed = self.preprocess(image)?;
        let (patches_x, patches_y) = self.patches_per_axis();
        let channels = 3usize;
        let patch_area = self.patch_size * self.patch_size;
        let patch_stride = self.patch_size * channels;
        let row_stride = self.preprocessor.target_width * channels;
        let mut embedding = vec![0.0f32; patches_x * patches_y * self.embedding_dim];

        for patch_y in 0..patches_y {
            for patch_x in 0..patches_x {
                let patch_idx = patch_y * patches_x + patch_x;
                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;

                for py in 0..self.patch_size {
                    for px in 0..self.patch_size {
                        let base = (patch_y * self.patch_size + py) * row_stride
                            + (patch_x * self.patch_size + px) * channels;
                        sum_r += preprocessed[base];
                        sum_g += preprocessed[base + 1];
                        sum_b += preprocessed[base + 2];
                    }
                }

                let denom = patch_area as f32;
                let mean_r = sum_r / denom;
                let mean_g = sum_g / denom;
                let mean_b = sum_b / denom;
                let patch_bias = patch_idx as f32 / (patches_x * patches_y).max(1) as f32;
                let base = patch_idx * self.embedding_dim;

                for dim in 0..self.embedding_dim {
                    let channel_mix = match dim % 3 {
                        0 => mean_r,
                        1 => mean_g,
                        _ => mean_b,
                    };
                    let position =
                        patch_bias + (dim as f32 / self.embedding_dim.max(1) as f32) * 0.1;
                    embedding[base + dim] = channel_mix + position;
                }
            }
        }

        Tensor::new(
            Some("vision_embeddings".to_string()),
            TensorType::F32,
            Shape::new(vec![patches_x * patches_y, self.embedding_dim]),
            TensorData::F32(embedding),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_resize_and_normalize() {
        let image = ImageInput::solid_rgb(2, 2, [255, 0, 0]);
        let resized = image.resize_nearest(4, 4).unwrap();
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        let normalized = resized.normalize();
        assert_eq!(normalized.len(), 4 * 4 * 3);
        assert!((normalized[0] - 1.0).abs() < f32::EPSILON);
        assert!((normalized[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_clip_encoder_shapes() {
        let encoder = ClipVisionEncoder::new((4, 4), 2, 8);
        let image = ImageInput::solid_rgb(2, 2, [64, 128, 255]);
        let embeddings = encoder.encode(&image).unwrap();
        assert_eq!(embeddings.shape().dims(), &[4, 8]);
    }
}
