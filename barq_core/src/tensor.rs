//! Tensor implementation with support for multiple data types and operations

use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Maximum number of dimensions in a tensor
pub const MAX_DIMS: usize = 4;

/// Default alignment for tensor data
pub const DEFAULT_ALIGNMENT: usize = 32;

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// Brain floating point (16-bit)
    Bf16,
    /// 64-bit floating point
    F64,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// Boolean
    Bool,
}

impl TensorType {
    /// Returns the size in bytes of this type
    #[inline]
    pub const fn size(&self) -> usize {
        match self {
            TensorType::F32 | TensorType::I32 | TensorType::U32 => 4,
            TensorType::F16 | TensorType::Bf16 | TensorType::I16 | TensorType::U16 => 2,
            TensorType::F64 | TensorType::I64 | TensorType::U64 => 8,
            TensorType::I8 | TensorType::U8 | TensorType::Bool => 1,
        }
    }

    /// Returns true if this is a floating point type
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(self, TensorType::F32 | TensorType::F16 | TensorType::Bf16 | TensorType::F64)
    }

    /// Returns true if this is an integer type
    #[inline]
    pub const fn is_int(&self) -> bool {
        matches!(
            self,
            TensorType::I8 | TensorType::I16 | TensorType::I32 | TensorType::I64
        )
    }

    /// Returns true if this is an unsigned integer type
    #[inline]
    pub const fn is_uint(&self) -> bool {
        matches!(
            self,
            TensorType::U8 | TensorType::U16 | TensorType::U32 | TensorType::U64
        )
    }

    /// Returns the name of this type
    pub fn name(&self) -> &'static str {
        match self {
            TensorType::F32 => "f32",
            TensorType::F16 => "f16",
            TensorType::Bf16 => "bf16",
            TensorType::F64 => "f64",
            TensorType::I8 => "i8",
            TensorType::I16 => "i16",
            TensorType::I32 => "i32",
            TensorType::I64 => "i64",
            TensorType::U8 => "u8",
            TensorType::U16 => "u16",
            TensorType::U32 => "u32",
            TensorType::U64 => "u64",
            TensorType::Bool => "bool",
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Tensor shape
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: Vec<usize>) -> Self {
        assert!(!dims.is_empty() && dims.len() <= MAX_DIMS, "Invalid shape dimensions");
        assert!(dims.iter().all(|&d| d > 0), "Invalid shape: dimensions must be positive");
        Self { dims }
    }

    /// Create a scalar shape (0-dimensional)
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Create a 1D shape
    pub fn vector(n: usize) -> Self {
        Self::new(vec![n])
    }

    /// Create a 2D shape (matrix)
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self::new(vec![rows, cols])
    }

    /// Returns the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns the dimensions
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the size of a specific dimension
    #[inline]
    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.dims.get(axis).copied()
    }

    /// Returns true if shapes are compatible for broadcasting
    pub fn is_broadcastable_to(&self, other: &Shape) -> bool {
        let ndim = self.ndim().max(other.ndim());
        for i in 0..ndim {
            let dim1 = if i < self.ndim() {
                self.dims[self.ndim() - i - 1]
            } else {
                1
            };
            let dim2 = if i < other.ndim() {
                other.dims[other.ndim() - i - 1]
            } else {
                1
            };
            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", itertools::join(self.dims.iter(), ", "))
    }
}

/// Tensor strides for non-contiguous memory
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides {
    bytes: Vec<usize>,
}

impl Strides {
    /// Create contiguous strides for a given shape and type size
    pub fn contiguous(shape: &Shape, type_size: usize) -> Self {
        let mut bytes = vec![type_size; shape.ndim()];
        for i in (1..shape.ndim()).rev() {
            bytes[i - 1] = bytes[i] * shape.dims()[i];
        }
        Self { bytes }
    }

    /// Returns the byte stride for a given dimension
    #[inline]
    pub fn stride(&self, dim: usize) -> usize {
        self.bytes[dim]
    }
}

/// Tensor data container
#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    F16(Vec<f16>),
    Bf16(Vec<bf16>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
}

impl TensorData {
    /// Returns the type of this data
    pub fn dtype(&self) -> TensorType {
        match self {
            TensorData::F32(_) => TensorType::F32,
            TensorData::F16(_) => TensorType::F16,
            TensorData::Bf16(_) => TensorType::Bf16,
            TensorData::F64(_) => TensorType::F64,
            TensorData::I8(_) => TensorType::I8,
            TensorData::I16(_) => TensorType::I16,
            TensorData::I32(_) => TensorType::I32,
            TensorData::I64(_) => TensorType::I64,
            TensorData::U8(_) => TensorType::U8,
            TensorData::U16(_) => TensorType::U16,
            TensorData::U32(_) => TensorType::U32,
            TensorData::U64(_) => TensorType::U64,
            TensorData::Bool(_) => TensorType::Bool,
        }
    }

    /// Returns the number of elements
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::F16(v) => v.len(),
            TensorData::Bf16(v) => v.len(),
            TensorData::F64(v) => v.len(),
            TensorData::I8(v) => v.len(),
            TensorData::I16(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I64(v) => v.len(),
            TensorData::U8(v) => v.len(),
            TensorData::U16(v) => v.len(),
            TensorData::U32(v) => v.len(),
            TensorData::U64(v) => v.len(),
            TensorData::Bool(v) => v.len(),
        }
    }

    /// Returns true if the data is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the size in bytes
    pub fn byte_size(&self) -> usize {
        self.len() * self.dtype().size()
    }
}

/// Tensor with reference-counted data
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name (optional)
    name: Option<String>,
    /// Data type
    dtype: TensorType,
    /// Shape
    shape: Shape,
    /// Byte strides
    strides: Strides,
    /// Tensor data
    data: Arc<TensorData>,
}

impl Tensor {
    /// Create a new tensor
    pub fn new(name: Option<String>, dtype: TensorType, shape: Shape, data: TensorData) -> Result<Self> {
        if data.len() != shape.num_elements() {
            return Err(Error::invalid_shape(shape.dims().to_vec()));
        }

        if data.dtype() != dtype {
            return Err(Error::type_mismatch(dtype.name(), data.dtype().name()));
        }

        let strides = Strides::contiguous(&shape, dtype.size());

        Ok(Self {
            name,
            dtype,
            shape,
            strides,
            data: Arc::new(data),
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(dtype: TensorType, shape: Shape) -> Self {
        let data = match dtype {
            TensorType::F32 => TensorData::F32(vec![0.0f32; shape.num_elements()]),
            TensorType::F16 => TensorData::F16(vec![f16::ZERO; shape.num_elements()]),
            TensorType::Bf16 => TensorData::Bf16(vec![bf16::ZERO; shape.num_elements()]),
            TensorType::F64 => TensorData::F64(vec![0.0f64; shape.num_elements()]),
            TensorType::I8 => TensorData::I8(vec![0i8; shape.num_elements()]),
            TensorType::I16 => TensorData::I16(vec![0i16; shape.num_elements()]),
            TensorType::I32 => TensorData::I32(vec![0i32; shape.num_elements()]),
            TensorType::I64 => TensorData::I64(vec![0i64; shape.num_elements()]),
            TensorType::U8 => TensorData::U8(vec![0u8; shape.num_elements()]),
            TensorType::U16 => TensorData::U16(vec![0u16; shape.num_elements()]),
            TensorType::U32 => TensorData::U32(vec![0u32; shape.num_elements()]),
            TensorType::U64 => TensorData::U64(vec![0u64; shape.num_elements()]),
            TensorType::Bool => TensorData::Bool(vec![false; shape.num_elements()]),
        };

        Self::new(None, dtype, shape, data).unwrap()
    }

    /// Create a tensor filled with ones
    pub fn ones(dtype: TensorType, shape: Shape) -> Self {
        let data = match dtype {
            TensorType::F32 => TensorData::F32(vec![1.0f32; shape.num_elements()]),
            TensorType::F16 => TensorData::F16(vec![f16::ONE; shape.num_elements()]),
            TensorType::Bf16 => TensorData::Bf16(vec![bf16::ONE; shape.num_elements()]),
            TensorType::F64 => TensorData::F64(vec![1.0f64; shape.num_elements()]),
            TensorType::I8 => TensorData::I8(vec![1i8; shape.num_elements()]),
            TensorType::I16 => TensorData::I16(vec![1i16; shape.num_elements()]),
            TensorType::I32 => TensorData::I32(vec![1i32; shape.num_elements()]),
            TensorType::I64 => TensorData::I64(vec![1i64; shape.num_elements()]),
            TensorType::U8 => TensorData::U8(vec![1u8; shape.num_elements()]),
            TensorType::U16 => TensorData::U16(vec![1u16; shape.num_elements()]),
            TensorType::U32 => TensorData::U32(vec![1u32; shape.num_elements()]),
            TensorType::U64 => TensorData::U64(vec![1u64; shape.num_elements()]),
            TensorType::Bool => TensorData::Bool(vec![true; shape.num_elements()]),
        };

        Self::new(None, dtype, shape, data).unwrap()
    }

    /// Returns the tensor name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the data type
    pub fn dtype(&self) -> TensorType {
        self.dtype
    }

    /// Returns the shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Returns the size in bytes
    pub fn byte_size(&self) -> usize {
        self.data.byte_size()
    }

    /// Returns true if this is a scalar
    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    /// Returns true if this is a vector (1D)
    pub fn is_vector(&self) -> bool {
        self.ndim() == 1
    }

    /// Returns true if this is a matrix (2D)
    pub fn is_matrix(&self) -> bool {
        self.ndim() == 2
    }

    /// Convert to f32 tensor
    pub fn to_f32(&self) -> Result<Tensor> {
        if self.dtype == TensorType::F32 {
            return Ok(self.clone());
        }

        let data = match &*self.data {
            TensorData::F16(v) => TensorData::F32(v.iter().map(|x| x.to_f32()).collect()),
            TensorData::Bf16(v) => TensorData::F32(v.iter().map(|x| x.to_f32()).collect()),
            TensorData::F64(v) => TensorData::F32(v.iter().map(|&x| x as f32).collect()),
            TensorData::I32(v) => TensorData::F32(v.iter().map(|&x| x as f32).collect()),
            _ => return Err(Error::type_mismatch("f32", self.dtype.name())),
        };

        Tensor::new(self.name.clone(), TensorType::F32, self.shape.clone(), data)
    }

    /// Get data as f32 slice (returns error if not f32)
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        match &*self.data {
            TensorData::F32(v) => Ok(v),
            _ => Err(Error::type_mismatch("f32", self.dtype.name())),
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "Tensor('{}', dtype={}, shape={})", name, self.dtype, self.shape)
        } else {
            write!(f, "Tensor(dtype={}, shape={})", self.dtype, self.shape)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let s = Shape::vector(10);
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.num_elements(), 10);

        let s = Shape::matrix(3, 4);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.num_elements(), 12);
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(TensorType::F32, Shape::matrix(2, 3));
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(tensor.dtype(), TensorType::F32);
    }
}
