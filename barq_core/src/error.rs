//! Error types for Barq core library

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Barq core library
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Tensor operation error
    #[error("Tensor error: {0}")]
    Tensor(String),

    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    Allocation(String),

    /// Invalid tensor shape
    #[error("Invalid tensor shape: {0:?}")]
    InvalidShape(Vec<usize>),

    /// Type mismatch
    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        expected: String,
        found: String,
    },

    /// Dimension mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid GGUF format
    #[error("Invalid GGUF format: {0}")]
    InvalidGguf(String),

    /// Quantization error
    #[error("Quantization error: {0}")]
    Quantization(String),

    /// Backend error
    #[error("Backend error: {0}")]
    Backend(String),

    /// Out of bounds access
    #[error("Index out of bounds: {index} in shape {shape:?}")]
    OutOfBounds {
        index: usize,
        shape: Vec<usize>,
    },

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

impl Error {
    pub fn tensor<S: Into<String>>(msg: S) -> Self {
        Error::Tensor(msg.into())
    }

    pub fn allocation<S: Into<String>>(msg: S) -> Self {
        Error::Allocation(msg.into())
    }

    pub fn invalid_shape(shape: Vec<usize>) -> Self {
        Error::InvalidShape(shape)
    }

    pub fn type_mismatch<E: Into<String>, F: Into<String>>(expected: E, found: F) -> Self {
        Error::TypeMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }

    pub fn dimension_mismatch<S: Into<String>>(msg: S) -> Self {
        Error::DimensionMismatch(msg.into())
    }

    pub fn quantization<S: Into<String>>(msg: S) -> Self {
        Error::Quantization(msg.into())
    }

    pub fn backend<S: Into<String>>(msg: S) -> Self {
        Error::Backend(msg.into())
    }
}
