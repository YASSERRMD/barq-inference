//! GGUF (GPT-Generated Unified Format) file implementation
//!
//! GGUF is a binary file format for storing LLM models efficiently.
//!
//! File structure:
//! 1. Magic number "GGUF" (4 bytes)
//! 2. Version (uint32)
//! 3. Tensor count (uint64)
//! 4. KV pair count (uint64)
//! 5. KV pairs
//! 6. Tensor info
//! 7. Tensor data

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian as LE, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::tensor::{Tensor, TensorType, Shape};
use crate::quant::QuantizationType;

/// GGUF magic number
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// GGUF version
pub const GGUF_VERSION: u32 = 3;

/// Default alignment
pub const DEFAULT_ALIGNMENT: u32 = 32;

/// GGUF value type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    /// Parse from u32
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GgufType::Uint8),
            1 => Ok(GgufType::Int8),
            2 => Ok(GgufType::Uint16),
            3 => Ok(GgufType::Int16),
            4 => Ok(GgufType::Uint32),
            5 => Ok(GgufType::Int32),
            6 => Ok(GgufType::Float32),
            7 => Ok(GgufType::Bool),
            8 => Ok(GgufType::String),
            9 => Ok(GgufType::Array),
            10 => Ok(GgufType::Uint64),
            11 => Ok(GgufType::Int64),
            12 => Ok(GgufType::Float64),
            _ => Err(Error::InvalidGguf(format!("Unknown type: {}", value))),
        }
    }
}

/// GGUF value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Get the type
    pub fn get_type(&self) -> GgufType {
        match self {
            GgufValue::Uint8(_) => GgufType::Uint8,
            GgufValue::Int8(_) => GgufType::Int8,
            GgufValue::Uint16(_) => GgufType::Uint16,
            GgufValue::Int16(_) => GgufType::Int16,
            GgufValue::Uint32(_) => GgufType::Uint32,
            GgufValue::Int32(_) => GgufType::Int32,
            GgufValue::Uint64(_) => GgufType::Uint64,
            GgufValue::Int64(_) => GgufType::Int64,
            GgufValue::Float32(_) => GgufType::Float32,
            GgufValue::Float64(_) => GgufType::Float64,
            GgufValue::Bool(_) => GgufType::Bool,
            GgufValue::String(_) => GgufType::String,
            GgufValue::Array(_) => GgufType::Array,
        }
    }
}

/// Tensor information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Tensor dimensions
    pub dimensions: Vec<u64>,
    /// Tensor data type
    pub dtype: TensorType,
    /// Offset in file
    pub offset: u64,
}

/// GGUF file reader
pub struct GgufReader {
    /// File reader
    reader: BufReader<File>,
    /// GGUF version
    version: u32,
    /// Number of tensors
    tensor_count: u64,
    /// Number of KV pairs
    kv_count: u64,
    /// Key-value pairs
    kv_pairs: HashMap<String, GgufValue>,
    /// Tensor information
    tensor_info: Vec<TensorInfo>,
    /// Alignment
    alignment: u32,
}

impl GgufReader {
    /// Open a GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Io(e))?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| Error::Io(e))?;
        if magic != *GGUF_MAGIC {
            return Err(Error::InvalidGguf(format!(
                "Invalid magic: {:?}, expected {:?}",
                magic, GGUF_MAGIC
            )));
        }

        // Read version
        let version = reader.read_u32::<LE>().map_err(|e| Error::Io(e))?;
        if version > GGUF_VERSION {
            return Err(Error::InvalidGguf(format!(
                "Unsupported version: {}, maximum supported: {}",
                version, GGUF_VERSION
            )));
        }

        // Read tensor count and KV count
        let tensor_count = reader.read_u64::<LE>().map_err(|e| Error::Io(e))?;
        let kv_count = reader.read_u64::<LE>().map_err(|e| Error::Io(e))?;

        // Read KV pairs
        let mut kv_pairs = HashMap::new();
        for _ in 0..kv_count {
            let key = Self::read_string(&mut reader)?;
            let value_type = GgufType::from_u32(reader.read_u32::<LE>().map_err(|e| Error::Io(e))?)?;
            let value = Self::read_value(&mut reader, value_type)?;
            kv_pairs.insert(key, value);
        }

        // Get alignment
        let alignment = match kv_pairs.get("general.alignment") {
            Some(GgufValue::Uint32(a)) => *a,
            _ => DEFAULT_ALIGNMENT,
        };

        // Read tensor info
        let mut tensor_info = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = Self::read_string(&mut reader)?;
            let n_dims = reader.read_u32::<LE>().map_err(|e| Error::Io(e))?;

            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64::<LE>().map_err(|e| Error::Io(e))?);
            }

            let dtype = TensorType::from_u32(reader.read_u32::<LE>().map_err(|e| Error::Io(e))?)?;
            let offset = reader.read_u64::<LE>().map_err(|e| Error::Io(e))?;

            tensor_info.push(TensorInfo {
                name,
                dimensions,
                dtype,
                offset,
            });
        }

        Ok(Self {
            reader,
            version,
            tensor_count,
            kv_count,
            kv_pairs,
            tensor_info,
            alignment,
        })
    }

    /// Read a string
    fn read_string<R: Read>(reader: &mut R) -> Result<String> {
        let len = reader.read_u64::<LE>().map_err(|e| Error::Io(e))? as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf).map_err(|e| Error::Io(e))?;
        String::from_utf8(buf).map_err(|e| Error::InvalidGguf(format!("Invalid UTF-8: {}", e)))
    }

    /// Read a value
    fn read_value<R: Read>(reader: &mut R, value_type: GgufType) -> Result<GgufValue> {
        Ok(match value_type {
            GgufType::Uint8 => GgufValue::Uint8(reader.read_u8().map_err(|e| Error::Io(e))?),
            GgufType::Int8 => GgufValue::Int8(reader.read_i8().map_err(|e| Error::Io(e))?),
            GgufType::Uint16 => GgufValue::Uint16(reader.read_u16::<LE>().map_err(|e| Error::Io(e))?),
            GgufType::Int16 => GgufValue::Int16(reader.read_i16::<LE>().map_err(|e| Error::Io(e))?),
            GgufType::Uint32 => GgufValue::Uint32(reader.read_u32::<LE>().map_err(|e| Error::Io(e))?),
            GgufType::Int32 => GgufValue::Int32(reader.read_i32::<LE>().map_err(|e| Error::Io(e))?),
            GgufType::Uint64 => GgufValue::Uint64(reader.read_u64::<LE>().map_err(|e| Error::Io(e))?),
            GgufType::Int64 => GgufValue::Int64(reader.read_i64::<LE>().map_err(|e| Error::Io(e))?),
            GgufType::Float32 => GgufValue::Float32(reader.read_f32::<LE>().map_err(|e| Error::Io(e))?),
            GgufType::Float64 => GgufValue::Float64(reader.read_f64::<LE>().map_err(|e| Error::Io(e))?),
            GgufType::Bool => GgufValue::Bool(reader.read_u8().map_err(|e| Error::Io(e))? != 0),
            GgufType::String => GgufValue::String(Self::read_string(reader)?),
            GgufType::Array => {
                let array_type = GgufType::from_u32(reader.read_u32::<LE>().map_err(|e| Error::Io(e))?)?;
                let len = reader.read_u64::<LE>().map_err(|e| Error::Io(e))? as usize;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(Self::read_value(reader, array_type)?);
                }
                GgufValue::Array(values)
            }
        })
    }

    /// Get the GGUF version
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Get the number of KV pairs
    pub fn kv_count(&self) -> u64 {
        self.kv_count
    }

    /// Get a KV value
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kv_pairs.get(key)
    }

    /// Get all KV pairs
    pub fn kv_pairs(&self) -> &HashMap<String, GgufValue> {
        &self.kv_pairs
    }

    /// Get tensor info
    pub fn tensor_info(&self) -> &[TensorInfo] {
        &self.tensor_info
    }

    /// Get tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_info.iter().find(|t| t.name == name)
    }

    /// Load a specific tensor
    pub fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        let info = self.get_tensor_info(name)
            .ok_or_else(|| Error::Tensor(format!("Tensor not found: {}", name)))?;

        // Seek to tensor data offset
        use std::io::Seek;
        self.reader.seek(io::SeekFrom::Start(info.offset))
            .map_err(|e| Error::Io(e))?;

        // Calculate total size
        let total_elements: usize = info.dimensions.iter().map(|&d| d as usize).product();
        let type_size = info.dtype.size();
        let total_bytes = total_elements * type_size;

        // Read tensor data
        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(|e| Error::Io(e))?;

        // Convert to TensorData based on type
        let tensor_data = match info.dtype {
            TensorType::F32 => {
                let values: Vec<f32> = data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                crate::tensor::TensorData::F32(values)
            }
            _ => return Err(Error::Unsupported(format!("Loading {} tensors", info.dtype))),
        };

        let shape = Shape::new(info.dimensions.iter().map(|&d| d as usize).collect());

        Tensor::new(Some(info.name.clone()), info.dtype, shape, tensor_data)
    }
}

impl TensorType {
    /// Parse from u32 (GGUF type ID)
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(TensorType::F32),
            1 => Ok(TensorType::F16),
            30 => Ok(TensorType::Bf16),
            28 => Ok(TensorType::F64),
            24 => Ok(TensorType::I8),
            25 => Ok(TensorType::I16),
            26 => Ok(TensorType::I32),
            27 => Ok(TensorType::I64),
            _ => Err(Error::InvalidGguf(format!("Unknown tensor type: {}", value))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_type() {
        assert_eq!(GgufType::from_u32(0).unwrap(), GgufType::Uint8);
        assert_eq!(GgufType::from_u32(8).unwrap(), GgufType::String);
        assert!(GgufType::from_u32(99).is_err());
    }
}
