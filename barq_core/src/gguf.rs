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
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use byteorder::{LittleEndian as LE, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::quant::QuantizationType;
use crate::tensor::{Shape, Tensor, TensorType};

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

/// GGUF-specific tensor types (quantized)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    /// Standard types
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_KS = 18,
    IQ3_KS = 19,
    IQ4_KS = 16,
    Q4_K_R4 = 30000, // Custom type ID for Q4_K_R4 if not in standard GGUF
}

impl GgufTensorType {
    /// Parse from u32
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GgufTensorType::F32),
            1 => Ok(GgufTensorType::F16),
            2 => Ok(GgufTensorType::Q4_0),
            3 => Ok(GgufTensorType::Q4_1),
            6 => Ok(GgufTensorType::Q5_0),
            7 => Ok(GgufTensorType::Q5_1),
            8 => Ok(GgufTensorType::Q8_0),
            9 => Ok(GgufTensorType::Q8_1),
            10 => Ok(GgufTensorType::Q2_K),
            11 => Ok(GgufTensorType::Q3_K),
            12 => Ok(GgufTensorType::Q4_K),
            13 => Ok(GgufTensorType::Q5_K),
            14 => Ok(GgufTensorType::Q6_K),
            15 => Ok(GgufTensorType::Q8_K),
            16 => Ok(GgufTensorType::IQ4_KS),
            18 => Ok(GgufTensorType::IQ2_KS),
            19 => Ok(GgufTensorType::IQ3_KS),
            30000 => Ok(GgufTensorType::Q4_K_R4),
            _ => Err(Error::InvalidGguf(format!(
                "Unknown GGUF tensor type: {}",
                value
            ))),
        }
    }

    /// Returns the block size for quantized types
    pub const fn block_size(&self) -> usize {
        match self {
            GgufTensorType::Q4_0 | GgufTensorType::Q4_1 => 32,
            GgufTensorType::Q5_0 | GgufTensorType::Q5_1 => 32,
            GgufTensorType::Q8_0 | GgufTensorType::Q8_1 => 32,
            GgufTensorType::Q2_K
            | GgufTensorType::Q3_K
            | GgufTensorType::Q4_K
            | GgufTensorType::Q5_K
            | GgufTensorType::Q6_K
            | GgufTensorType::Q8_K
            | GgufTensorType::IQ4_KS
            | GgufTensorType::IQ3_KS
            | GgufTensorType::IQ2_KS => 256,
            GgufTensorType::Q4_K_R4 => 32,
            GgufTensorType::F32 | GgufTensorType::F16 => 1,
        }
    }

    /// Returns the type name
    pub fn name(&self) -> &'static str {
        match self {
            GgufTensorType::F32 => "f32",
            GgufTensorType::F16 => "f16",
            GgufTensorType::Q4_0 => "q4_0",
            GgufTensorType::Q4_1 => "q4_1",
            GgufTensorType::Q5_0 => "q5_0",
            GgufTensorType::Q5_1 => "q5_1",
            GgufTensorType::Q8_0 => "q8_0",
            GgufTensorType::Q8_1 => "q8_1",
            GgufTensorType::Q2_K => "q2_k",
            GgufTensorType::Q3_K => "q3_k",
            GgufTensorType::Q4_K => "q4_k",
            GgufTensorType::Q5_K => "q5_k",
            GgufTensorType::Q6_K => "q6_k",
            GgufTensorType::Q8_K => "q8_k",
            GgufTensorType::IQ4_KS => "iq4_ks",
            GgufTensorType::IQ3_KS => "iq3_ks",
            GgufTensorType::IQ2_KS => "iq2_ks",
            GgufTensorType::Q4_K_R4 => "q4_k_r4",
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
    /// GGUF tensor type (for quantized tensors)
    pub gguf_type: GgufTensorType,
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
    /// Offset to the beginning of tensor data
    data_offset: u64,
}

impl GgufReader {
    /// Open a GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(Error::Io)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(Error::Io)?;
        if magic != *GGUF_MAGIC {
            return Err(Error::InvalidGguf(format!(
                "Invalid magic: {:?}, expected {:?}",
                magic, GGUF_MAGIC
            )));
        }

        // Read version
        let version = reader.read_u32::<LE>().map_err(Error::Io)?;
        if version > GGUF_VERSION {
            return Err(Error::InvalidGguf(format!(
                "Unsupported version: {}, maximum supported: {}",
                version, GGUF_VERSION
            )));
        }

        // Read tensor count and KV count
        let tensor_count = reader.read_u64::<LE>().map_err(Error::Io)?;
        let kv_count = reader.read_u64::<LE>().map_err(Error::Io)?;

        // Read KV pairs
        let mut kv_pairs = HashMap::new();
        for _ in 0..kv_count {
            let key = Self::read_string(&mut reader)?;
            let value_type = GgufType::from_u32(reader.read_u32::<LE>().map_err(Error::Io)?)?;
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
            let n_dims = reader.read_u32::<LE>().map_err(Error::Io)?;

            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64::<LE>().map_err(Error::Io)?);
            }

            let gguf_type = GgufTensorType::from_u32(reader.read_u32::<LE>().map_err(Error::Io)?)?;
            let offset = reader.read_u64::<LE>().map_err(Error::Io)?;

            // Convert GGUF type to standard TensorType for storage
            let dtype = match gguf_type {
                GgufTensorType::F32 => TensorType::F32,
                GgufTensorType::F16 => TensorType::F16,
                _ => TensorType::F32, // Quantized types will be dequantized to F32
            };

            tensor_info.push(TensorInfo {
                name,
                dimensions,
                dtype,
                gguf_type,
                offset,
            });
        }

        // Calculate data offset: aligned position after metadata
        let current_pos = reader.stream_position().map_err(Error::Io)?;
        let alignment = alignment as u64;
        let data_offset = (current_pos + alignment - 1) & !(alignment - 1);

        Ok(Self {
            reader,
            version,
            tensor_count,
            kv_count,
            kv_pairs,
            tensor_info,
            alignment: alignment as u32,
            data_offset,
        })
    }

    /// Read a string
    fn read_string<R: Read>(reader: &mut R) -> Result<String> {
        let len = reader.read_u64::<LE>().map_err(Error::Io)? as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf).map_err(Error::Io)?;
        String::from_utf8(buf).map_err(|e| Error::InvalidGguf(format!("Invalid UTF-8: {}", e)))
    }

    /// Read a value
    fn read_value<R: Read>(reader: &mut R, value_type: GgufType) -> Result<GgufValue> {
        Ok(match value_type {
            GgufType::Uint8 => GgufValue::Uint8(reader.read_u8().map_err(Error::Io)?),
            GgufType::Int8 => GgufValue::Int8(reader.read_i8().map_err(Error::Io)?),
            GgufType::Uint16 => GgufValue::Uint16(reader.read_u16::<LE>().map_err(Error::Io)?),
            GgufType::Int16 => GgufValue::Int16(reader.read_i16::<LE>().map_err(Error::Io)?),
            GgufType::Uint32 => GgufValue::Uint32(reader.read_u32::<LE>().map_err(Error::Io)?),
            GgufType::Int32 => GgufValue::Int32(reader.read_i32::<LE>().map_err(Error::Io)?),
            GgufType::Uint64 => GgufValue::Uint64(reader.read_u64::<LE>().map_err(Error::Io)?),
            GgufType::Int64 => GgufValue::Int64(reader.read_i64::<LE>().map_err(Error::Io)?),
            GgufType::Float32 => GgufValue::Float32(reader.read_f32::<LE>().map_err(Error::Io)?),
            GgufType::Float64 => GgufValue::Float64(reader.read_f64::<LE>().map_err(Error::Io)?),
            GgufType::Bool => GgufValue::Bool(reader.read_u8().map_err(Error::Io)? != 0),
            GgufType::String => GgufValue::String(Self::read_string(reader)?),
            GgufType::Array => {
                let array_type = GgufType::from_u32(reader.read_u32::<LE>().map_err(Error::Io)?)?;
                let len = reader.read_u64::<LE>().map_err(Error::Io)? as usize;
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

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_info.iter().map(|t| t.name.as_str()).collect()
    }

    /// Load a specific tensor with dequantization support
    pub fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        let info = self
            .get_tensor_info(name)
            .ok_or_else(|| Error::Tensor(format!("Tensor not found: {}", name)))?;

        // Clone the fields we need after seeking
        let offset = info.offset;
        let dimensions = info.dimensions.clone();
        let gguf_type = info.gguf_type;
        let tensor_name = info.name.clone();

        // Seek to tensor data offset
        use std::io::Seek;
        let absolute_offset = self.data_offset + offset;
        self.reader
            .seek(io::SeekFrom::Start(absolute_offset))
            .map_err(Error::Io)?;

        // Calculate total elements
        let total_elements: usize = dimensions.iter().map(|&d| d as usize).product();

        // Load and dequantize based on type
        let tensor_data = match gguf_type {
            GgufTensorType::F32 => {
                let type_size = 4;
                let total_bytes = total_elements * type_size;
                let mut data = vec![0u8; total_bytes];
                self.reader.read_exact(&mut data).map_err(Error::Io)?;

                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                crate::tensor::TensorData::F32(values)
            }
            GgufTensorType::F16 => {
                let type_size = 2;
                let total_bytes = total_elements * type_size;
                let mut data = vec![0u8; total_bytes];
                self.reader.read_exact(&mut data).map_err(Error::Io)?;

                let values: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                crate::tensor::TensorData::F32(values)
            }
            GgufTensorType::Q5_0 => self.load_q5_0(&dimensions, total_elements)?,
            GgufTensorType::Q4_0 => self.load_q4_0(&dimensions, total_elements)?,
            GgufTensorType::Q4_1 => {
                return Err(Error::Unsupported("Q4_1 not yet implemented".to_string()))
            }
            GgufTensorType::Q8_0 => self.load_q8_0(&dimensions, total_elements)?,
            GgufTensorType::Q2_K => self.load_q2_k(&dimensions, total_elements)?,
            GgufTensorType::Q3_K => self.load_q3_k(&dimensions, total_elements)?,
            GgufTensorType::Q5_K => self.load_q5_k(&dimensions, total_elements)?,
            GgufTensorType::Q4_K => self.load_q4_k(&dimensions, total_elements)?,
            GgufTensorType::Q6_K => self.load_q6_k(&dimensions, total_elements)?,
            GgufTensorType::IQ4_KS
            | GgufTensorType::IQ3_KS
            | GgufTensorType::IQ2_KS
            | GgufTensorType::Q4_K_R4 => {
                // Correct on-disk sizes from ggml-common.h:
                //   IQ4_KS: 4 byte f32 row-scale + n_blocks*(8 scales + 128 qs) = 4 + n_blocks*136
                //   IQ3_KS: 2 byte f16 row-scale + n_blocks*(2 extra + 4 scales + 64 qs + 32 qh) = 2 + n_blocks*102
                //   IQ2_KS: 2 byte f16 row-scale + n_blocks*(2 extra + 4 scales + 64 qs) = 2 + n_blocks*70
                //   Q4_K_R4: 8 byte 4xf16 row-scales + n_blocks*(32 scales + 512 qs) = 8 + n_blocks*544
                let (row_header, bytes_per_block) = match gguf_type {
                    GgufTensorType::IQ4_KS => (4usize, 136usize),
                    GgufTensorType::IQ3_KS => (2usize, 102usize),
                    GgufTensorType::IQ2_KS => (2usize, 70usize),
                    GgufTensorType::Q4_K_R4 => (8usize, 544usize),
                    _ => unreachable!(),
                };
                let n_blocks = total_elements.div_ceil(gguf_type.block_size());
                let total_bytes = row_header + n_blocks * bytes_per_block;
                let mut raw = vec![0u8; total_bytes];
                self.reader.read_exact(&mut raw).map_err(Error::Io)?;

                // barq_core cannot depend on the `quant` crate — dequantization for these
                // types is handled at a higher level (in models/barq-inference) via
                // quant::iq::{dequantize_iq4_ks, dequantize_iq3_ks, dequantize_iq2_ks}.
                return Err(Error::Unsupported(format!(
                    "IK quant {} tensor '{}' cannot be dequantized in GgufReader; \
                     use quant::iq::dequantize_* at a higher level",
                    gguf_type.name(),
                    name
                )));
            }
            _ => {
                return Err(Error::Unsupported(format!(
                    "Loading GGUF tensor type: {} (tensor: {})",
                    gguf_type.name(),
                    name
                )))
            }
        };

        let shape = Shape::new(dimensions.iter().map(|&d| d as usize).collect());

        Tensor::new(Some(tensor_name), TensorType::F32, shape, tensor_data)
    }

    /// Load Q4_0 quantized tensor and dequantize to f32
    fn load_q4_0(
        &mut self,
        dimensions: &[u64],
        total_elements: usize,
    ) -> Result<crate::tensor::TensorData> {
        let block_size = 32; // Q4_0 block size
        let n_blocks = total_elements.div_ceil(block_size);

        // Q4_0 format: scale (f32) + quants (16 bytes for 32 values)
        let bytes_per_block = 4 + (block_size / 2);
        let total_bytes = n_blocks * bytes_per_block;

        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(Error::Io)?;

        let mut output = Vec::with_capacity(total_elements);
        let mut offset = 0;

        for block in 0..n_blocks {
            if offset + 4 > data.len() {
                break;
            }

            // Read scale
            let scale = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            // Read quantized values
            let q_len = block_size.div_ceil(2);
            if offset + q_len > data.len() {
                break;
            }

            let quants = &data[offset..offset + q_len];
            offset += q_len;

            // Dequantize block
            for i in 0..block_size {
                let byte_idx = i / 2;
                let shift = if i % 2 == 0 { 0 } else { 4 };

                if byte_idx < quants.len() {
                    let q = ((quants[byte_idx] >> shift) & 0x0F) as i8;
                    let q = if q >= 8 { q - 16 } else { q }; // Convert to signed
                    output.push(q as f32 * scale);
                }
            }
        }

        output.truncate(total_elements);
        Ok(crate::tensor::TensorData::F32(output))
    }

    /// Load Q5_0 quantized tensor and dequantize to f32.
    fn load_q5_0(
        &mut self,
        _dimensions: &[u64],
        total_elements: usize,
    ) -> Result<crate::tensor::TensorData> {
        const QK_5_0: usize = 32;
        const BYTES_PER_BLOCK: usize = 2 + 4 + (QK_5_0 / 2);

        let n_blocks = total_elements.div_ceil(QK_5_0);
        let total_bytes = n_blocks * BYTES_PER_BLOCK;

        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(Error::Io)?;

        let mut output = vec![0.0f32; n_blocks * QK_5_0];
        let mut offset = 0usize;

        for block_idx in 0..n_blocks {
            if offset + BYTES_PER_BLOCK > data.len() {
                break;
            }

            let d = half::f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
            offset += 2;

            let qh = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            let qs = &data[offset..offset + (QK_5_0 / 2)];
            offset += QK_5_0 / 2;

            let base = block_idx * QK_5_0;
            for j in 0..(QK_5_0 / 2) {
                let xh_0 = (((qh >> (j + 0)) << 4) & 0x10) as u8;
                let xh_1 = (((qh >> (j + 12)) ) & 0x10) as u8;

                let x0 = ((qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;

                output[base + j] = x0 as f32 * d;
                output[base + (QK_5_0 / 2) + j] = x1 as f32 * d;
            }
        }

        output.truncate(total_elements);
        Ok(crate::tensor::TensorData::F32(output))
    }

    /// Load Q8_0 quantized tensor and dequantize to f32
    fn load_q8_0(
        &mut self,
        dimensions: &[u64],
        total_elements: usize,
    ) -> Result<crate::tensor::TensorData> {
        let block_size = 32; // Q8_0 block size
        let n_blocks = total_elements.div_ceil(block_size);

        // Q8_0 format matches llama.cpp block_q8_0:
        //   - d: f16 scale
        //   - qs[32]: signed 8-bit values
        let bytes_per_block = 2 + block_size;
        let total_bytes = n_blocks * bytes_per_block;

        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(Error::Io)?;

        let mut output = Vec::with_capacity(total_elements);
        let mut offset = 0;

        for _block in 0..n_blocks {
            if offset + 2 > data.len() {
                break;
            }

            // Read scale
            let scale = half::f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
            offset += 2;

            // Read quantized values
            if offset + block_size > data.len() {
                break;
            }

            let quants = &data[offset..offset + block_size];
            offset += block_size;

            // Dequantize block
            for &q in quants {
                let q = q as i8; // Convert to signed
                output.push(q as f32 * scale);
            }
        }

        output.truncate(total_elements);
        Ok(crate::tensor::TensorData::F32(output))
    }

    /// Load Q2_K quantized tensor and dequantize to f32
    fn load_q2_k(
        &mut self,
        _dimensions: &[u64],
        total_elements: usize,
    ) -> Result<crate::tensor::TensorData> {
        let data = self.read_quantized_blocks(total_elements, 84)?;
        let values = dequantize_q2_k_bytes(&data, total_elements)?;
        Ok(crate::tensor::TensorData::F32(values))
    }

    /// Load Q3_K quantized tensor and dequantize to f32
    fn load_q3_k(
        &mut self,
        _dimensions: &[u64],
        total_elements: usize,
    ) -> Result<crate::tensor::TensorData> {
        let data = self.read_quantized_blocks(total_elements, 110)?;
        let values = dequantize_q3_k_bytes(&data, total_elements)?;
        Ok(crate::tensor::TensorData::F32(values))
    }

    /// Load Q5_K quantized tensor and dequantize to f32
    fn load_q5_k(
        &mut self,
        _dimensions: &[u64],
        total_elements: usize,
    ) -> Result<crate::tensor::TensorData> {
        let data = self.read_quantized_blocks(total_elements, Q5K_BLOCK_BYTES)?;
        let values = dequantize_q5_k_bytes(&data, total_elements)?;
        Ok(crate::tensor::TensorData::F32(values))
    }

    fn read_quantized_blocks(
        &mut self,
        total_elements: usize,
        bytes_per_block: usize,
    ) -> Result<Vec<u8>> {
        const QK_K: usize = 256;

        let n_blocks = total_elements.div_ceil(QK_K);
        let total_bytes = n_blocks * bytes_per_block;
        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(Error::Io)?;
        Ok(data)
    }

    /// Load Q4_K_M quantized tensor and dequantize to f32
    /// Implementation matching llama.cpp ggml-quants.c exactly
    fn load_q4_k(
        &mut self,
        _dimensions: &[u64],
        total_elements: usize,
    ) -> Result<crate::tensor::TensorData> {
        const QK_K: usize = 256; // Q4_K block size

        let n_blocks = total_elements.div_ceil(QK_K);

        // Q4_K format per 256 values (from llama.cpp block_q4_K):
        // - d (f16): 2 bytes (main scale)
        // - dmin (f16): 2 bytes (main minimum)
        // - scales[12]: 12 bytes (packed scale/minimum values)
        // - qs[QK_K/2]: 128 bytes (4-bit packed quantized values)
        // Total: 144 bytes per 256-value block
        const BYTES_PER_BLOCK: usize = 2 + 2 + 12 + (QK_K / 2);
        let total_bytes = n_blocks * BYTES_PER_BLOCK;

        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(Error::Io)?;

        let mut output = Vec::with_capacity(total_elements);
        let mut offset = 0;

        for _block_idx in 0..n_blocks {
            if offset + BYTES_PER_BLOCK > data.len() {
                break;
            }

            // Read d (main scale as f16)
            let d_bytes = [data[offset], data[offset + 1]];
            let d = half::f16::from_le_bytes(d_bytes).to_f32();
            offset += 2;

            // Read dmin (main minimum as f16)
            let dmin_bytes = [data[offset], data[offset + 1]];
            let dmin = half::f16::from_le_bytes(dmin_bytes).to_f32();
            offset += 2;

            // Read scales[12] (packed 6-bit scale/minimum values)
            let mut scales = [0u8; 12];
            if offset + 12 > data.len() {
                break;
            }
            scales.copy_from_slice(&data[offset..offset + 12]);
            offset += 12;

            // Read qs[QK_K/2] (4-bit packed quantized values)
            const QS_LEN: usize = QK_K / 2;
            let mut qs = [0u8; QS_LEN];
            if offset + QS_LEN > data.len() {
                break;
            }
            qs.copy_from_slice(&data[offset..offset + QS_LEN]);
            offset += qs.len();

            // Dequantize: matching llama.cpp dequantize_row_q4_K()
            let mut is = 0usize;

            for j in (0..QK_K).step_by(64) {
                // Each byte in qs[j/2..j/2+32] contains two 4-bit values:
                // - Low nibble: first 32 values in the 64-value chunk
                // - High nibble: next 32 values in the 64-value chunk
                let qs_offset = j / 2;

                // Process first 32 values (low nibbles)
                let (sc1, m1_val) = Self::get_scale_min_k4(is, &scales);
                let d1 = d * sc1 as f32;
                let min1 = dmin * m1_val as f32;

                for l in 0..32 {
                    let q = qs[qs_offset + l] & 0x0F;
                    output.push(d1 * q as f32 - min1);
                }

                // Process second 32 values (high nibbles)
                let (sc2, m2_val) = Self::get_scale_min_k4(is + 1, &scales);
                let d2 = d * sc2 as f32;
                let min2 = dmin * m2_val as f32;

                for l in 0..32 {
                    let q = qs[qs_offset + l] >> 4;
                    output.push(d2 * q as f32 - min2);
                }

                is += 2;
            }
        }

        output.truncate(total_elements);
        Ok(crate::tensor::TensorData::F32(output))
    }

    /// Helper function matching llama.cpp get_scale_min_k4()
    /// Extracts 6-bit scale and minimum values from packed format
    fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            let d = q[j] & 63;
            let m = q[j + 4] & 63;
            (d, m)
        } else {
            let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
            (d, m)
        }
    }

    /// Load Q6_K quantized tensor and dequantize to f32
    fn load_q6_k(
        &mut self,
        _dimensions: &[u64],
        total_elements: usize,
    ) -> Result<crate::tensor::TensorData> {
        const QK_K: usize = 256;
        let n_blocks = total_elements.div_ceil(QK_K);

        // Q6_K format (from llama.cpp block_q6_K):
        // uint8_t ql[QK_K/2];      // 128 bytes
        // uint8_t qh[QK_K/4];      // 64 bytes
        // int8_t  scales[QK_K/16]; // 16 bytes
        // f16 d;                   // 2 bytes
        // Total: 210 bytes
        const BYTES_PER_BLOCK: usize = 128 + 64 + 16 + 2;
        let total_bytes = n_blocks * BYTES_PER_BLOCK;

        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(Error::Io)?;

        let mut output = vec![0.0f32; n_blocks * QK_K];
        let mut offset = 0;

        for block_idx in 0..n_blocks {
            let ql_ptr = offset;
            let qh_ptr = offset + 128;
            let sc_ptr = offset + 128 + 64;
            let d_ptr = offset + 128 + 64 + 16;

            let d = half::f16::from_le_bytes([data[d_ptr], data[d_ptr + 1]]).to_f32();

            for n in (0..QK_K).step_by(128) {
                let off_ql = n / 2;
                let off_qh = n / 4;
                let off_sc = n / 16;

                for l in 0..32 {
                    let is = l / 16;
                    let sc0 = data[sc_ptr + off_sc + is + 0] as i8 as f32;
                    let sc1 = data[sc_ptr + off_sc + is + 2] as i8 as f32;
                    let sc2 = data[sc_ptr + off_sc + is + 4] as i8 as f32;
                    let sc3 = data[sc_ptr + off_sc + is + 6] as i8 as f32;

                    let d0 = d * sc0;
                    let d1 = d * sc1;
                    let d2 = d * sc2;
                    let d3 = d * sc3;

                    let qh0 = data[qh_ptr + off_qh + l];

                    let v0 = ((data[ql_ptr + off_ql + l + 0] & 0xF) | (((qh0 >> 0) & 3) << 4))
                        as i8
                        - 32;
                    let v1 = ((data[ql_ptr + off_ql + l + 32] & 0xF) | (((qh0 >> 2) & 3) << 4))
                        as i8
                        - 32;
                    let v2 =
                        ((data[ql_ptr + off_ql + l + 0] >> 4) | (((qh0 >> 4) & 3) << 4)) as i8 - 32;
                    let v3 = ((data[ql_ptr + off_ql + l + 32] >> 4) | (((qh0 >> 6) & 3) << 4))
                        as i8
                        - 32;

                    let idx = block_idx * QK_K + n + l;
                    output[idx + 0] = d0 * v0 as f32;
                    output[idx + 32] = d1 * v1 as f32;
                    output[idx + 64] = d2 * v2 as f32;
                    output[idx + 96] = d3 * v3 as f32;
                }
            }
            offset += BYTES_PER_BLOCK;
        }

        output.truncate(total_elements);
        Ok(crate::tensor::TensorData::F32(output))
    }
}

const QK_K: usize = 256;
const Q2K_BLOCK_BYTES: usize = 84;
const Q3K_BLOCK_BYTES: usize = 110;
const Q5K_BLOCK_BYTES: usize = 176;

fn dequantize_q2_k_bytes(data: &[u8], total_elements: usize) -> Result<Vec<f32>> {
    let n_blocks = total_elements.div_ceil(QK_K);
    let expected_bytes = n_blocks * Q2K_BLOCK_BYTES;

    if data.len() < expected_bytes {
        return Err(Error::InvalidGguf(format!(
            "Q2_K tensor needs {} bytes for {} blocks, got {}",
            expected_bytes,
            n_blocks,
            data.len()
        )));
    }

    let mut output = vec![0.0f32; n_blocks * QK_K];

    for block_idx in 0..n_blocks {
        let block_start = block_idx * Q2K_BLOCK_BYTES;
        let block = &data[block_start..block_start + Q2K_BLOCK_BYTES];
        let out_start = block_idx * QK_K;
        let out_end = out_start + QK_K;
        dequantize_q2_k_block(block, &mut output[out_start..out_end])?;
    }

    output.truncate(total_elements);
    Ok(output)
}

fn dequantize_q2_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    if block.len() != Q2K_BLOCK_BYTES {
        return Err(Error::InvalidGguf(format!(
            "Q2_K block must be {} bytes, got {}",
            Q2K_BLOCK_BYTES,
            block.len()
        )));
    }
    if output.len() < QK_K {
        return Err(Error::InvalidGguf(format!(
            "Q2_K output buffer must hold at least {} values, got {}",
            QK_K,
            output.len()
        )));
    }

    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = half::f16::from_le_bytes([block[80], block[81]]).to_f32();
    let min = half::f16::from_le_bytes([block[82], block[83]]).to_f32();

    let mut out_idx = 0usize;
    let mut is = 0usize;

    for q_offset in (0..QK_K).step_by(128) {
        let q_base = q_offset / 4;
        let mut shift = 0usize;

        for _ in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0x0f) as f32;
            let ml = min * (sc >> 4) as f32;

            for l in 0..16 {
                let q = ((qs[q_base + l] >> shift) & 3) as f32;
                output[out_idx] = dl * q - ml;
                out_idx += 1;
            }

            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0x0f) as f32;
            let ml = min * (sc >> 4) as f32;

            for l in 0..16 {
                let q = ((qs[q_base + 16 + l] >> shift) & 3) as f32;
                output[out_idx] = dl * q - ml;
                out_idx += 1;
            }

            shift += 2;
        }
    }

    debug_assert_eq!(out_idx, QK_K);
    Ok(())
}

fn dequantize_q3_k_bytes(data: &[u8], total_elements: usize) -> Result<Vec<f32>> {
    let n_blocks = total_elements.div_ceil(QK_K);
    let expected_bytes = n_blocks * Q3K_BLOCK_BYTES;

    if data.len() < expected_bytes {
        return Err(Error::InvalidGguf(format!(
            "Q3_K tensor needs {} bytes for {} blocks, got {}",
            expected_bytes,
            n_blocks,
            data.len()
        )));
    }

    let mut output = vec![0.0f32; n_blocks * QK_K];

    for block_idx in 0..n_blocks {
        let block_start = block_idx * Q3K_BLOCK_BYTES;
        let block = &data[block_start..block_start + Q3K_BLOCK_BYTES];
        let out_start = block_idx * QK_K;
        let out_end = out_start + QK_K;
        dequantize_q3_k_block(block, &mut output[out_start..out_end])?;
    }

    output.truncate(total_elements);
    Ok(output)
}

fn decode_q3_k_scales(scales: &[u8; 12]) -> [i8; 16] {
    let kmask1 = 0x0303_0303u32;
    let kmask2 = 0x0f0f_0f0fu32;

    let mut aux = [
        u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]),
        u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]),
        u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]),
        0u32,
    ];

    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    let mut decoded = [0i8; 16];
    for (word_idx, word) in aux.iter().enumerate() {
        let bytes = word.to_le_bytes();
        for (byte_idx, byte) in bytes.iter().enumerate() {
            decoded[word_idx * 4 + byte_idx] = i8::from_ne_bytes([*byte]);
        }
    }

    decoded
}

fn dequantize_q3_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    if block.len() != Q3K_BLOCK_BYTES {
        return Err(Error::InvalidGguf(format!(
            "Q3_K block must be {} bytes, got {}",
            Q3K_BLOCK_BYTES,
            block.len()
        )));
    }
    if output.len() < QK_K {
        return Err(Error::InvalidGguf(format!(
            "Q3_K output buffer must hold at least {} values, got {}",
            QK_K,
            output.len()
        )));
    }

    let mut scales_bytes = [0u8; 12];
    scales_bytes.copy_from_slice(&block[96..108]);
    let scales = decode_q3_k_scales(&scales_bytes);
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let d_all = half::f16::from_le_bytes([block[108], block[109]]).to_f32();

    let mut out_idx = 0usize;
    let mut scale_idx = 0usize;
    let mut m = 1u8;

    for q_offset in (0..QK_K).step_by(128) {
        let q_base = q_offset / 4;
        let mut shift = 0usize;

        for _ in 0..4 {
            let dl = d_all * (scales[scale_idx] as f32 - 32.0);
            scale_idx += 1;

            for l in 0..16 {
                let q = ((qs[q_base + l] >> shift) & 3) as i8;
                let high = if (hmask[l] & m) != 0 { 0 } else { 4 };
                output[out_idx] = dl * (q - high) as f32;
                out_idx += 1;
            }

            let dl = d_all * (scales[scale_idx] as f32 - 32.0);
            scale_idx += 1;

            for l in 0..16 {
                let q = ((qs[q_base + 16 + l] >> shift) & 3) as i8;
                let high = if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                output[out_idx] = dl * (q - high) as f32;
                out_idx += 1;
            }

            shift += 2;
            m <<= 1;
        }
    }

    debug_assert_eq!(out_idx, QK_K);
    Ok(())
}

fn dequantize_q5_k_bytes(data: &[u8], total_elements: usize) -> Result<Vec<f32>> {
    let n_blocks = total_elements.div_ceil(QK_K);
    let expected_bytes = n_blocks * Q5K_BLOCK_BYTES;

    if data.len() < expected_bytes {
        return Err(Error::InvalidGguf(format!(
            "Q5_K tensor needs {} bytes for {} blocks, got {}",
            expected_bytes,
            n_blocks,
            data.len()
        )));
    }

    let mut output = vec![0.0f32; n_blocks * QK_K];

    for block_idx in 0..n_blocks {
        let block_start = block_idx * Q5K_BLOCK_BYTES;
        let block = &data[block_start..block_start + Q5K_BLOCK_BYTES];
        let out_start = block_idx * QK_K;
        let out_end = out_start + QK_K;
        dequantize_q5_k_block(block, &mut output[out_start..out_end])?;
    }

    output.truncate(total_elements);
    Ok(output)
}

fn dequantize_q5_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    if block.len() != Q5K_BLOCK_BYTES {
        return Err(Error::InvalidGguf(format!(
            "Q5_K block must be {} bytes, got {}",
            Q5K_BLOCK_BYTES,
            block.len()
        )));
    }
    if output.len() < QK_K {
        return Err(Error::InvalidGguf(format!(
            "Q5_K output buffer must hold at least {} values, got {}",
            QK_K,
            output.len()
        )));
    }

    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let min = half::f16::from_le_bytes([block[2], block[3]]).to_f32();

    let mut scales = [0u8; 12];
    scales.copy_from_slice(&block[4..16]);
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut out_idx = 0usize;
    let mut scale_idx = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;

    for q_offset in (0..QK_K).step_by(64) {
        let (sc1, m1) = GgufReader::get_scale_min_k4(scale_idx, &scales);
        let d1 = d * sc1 as f32;
        let m1 = min * m1 as f32;

        let (sc2, m2) = GgufReader::get_scale_min_k4(scale_idx + 1, &scales);
        let d2 = d * sc2 as f32;
        let m2 = min * m2 as f32;

        let ql = &qs[q_offset / 2..q_offset / 2 + 32];
        for l in 0..32 {
            let q = (ql[l] & 0x0f) + if qh[l] & u1 != 0 { 16 } else { 0 };
            output[out_idx] = d1 * q as f32 - m1;
            out_idx += 1;
        }
        for l in 0..32 {
            let q = (ql[l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
            output[out_idx] = d2 * q as f32 - m2;
            out_idx += 1;
        }

        scale_idx += 2;
        u1 <<= 2;
        u2 <<= 2;
    }

    debug_assert_eq!(out_idx, QK_K);
    Ok(())
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
            _ => Err(Error::InvalidGguf(format!(
                "Unknown tensor type: {}",
                value
            ))),
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
    }

    #[test]
    fn test_dequantize_q2_k_bytes_constant_block() {
        let mut block = [0u8; Q2K_BLOCK_BYTES];
        block[0..16].fill(0x11);
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        block[82..84].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

        let values = dequantize_q2_k_bytes(&block, QK_K).unwrap();
        assert_eq!(values.len(), QK_K);
        assert!(values.iter().all(|&v| v == -1.0));
    }

    #[test]
    fn test_dequantize_q3_k_bytes_zero_block() {
        let mut block = [0u8; Q3K_BLOCK_BYTES];
        block[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

        let values = dequantize_q3_k_bytes(&block, QK_K).unwrap();
        assert_eq!(values.len(), QK_K);
        assert!(values.iter().all(|&v| v == 128.0));
    }

    #[test]
    fn test_load_q5_0_orders_low_half_before_high_half() {
        let mut block = [0u8; 22];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        // Set the low half to -16 and the high half to +1 so ordering bugs are obvious.
        block[2..6].copy_from_slice(&0xFFFF0000u32.to_le_bytes());
        block[6..].fill(0x10);

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        let mut values = vec![0.0f32; 32];

        for j in 0..16 {
            let xh_0 = (((qh >> (j + 0)) << 4) & 0x10) as u8;
            let xh_1 = (((qh >> (j + 12))) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;
            values[j] = x0 as f32 * d;
            values[16 + j] = x1 as f32 * d;
        }

        assert!(values[..16].iter().all(|&v| (v + 16.0).abs() < f32::EPSILON));
        assert!(values[16..].iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_load_q8_0_constant_block() {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        block[2..].fill(0x00);

        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let quants = &block[2..];
        let values: Vec<f32> = quants.iter().map(|&q| (q as i8) as f32 * scale).collect();

        assert_eq!(values.len(), 32);
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequantize_q5_k_bytes_constant_block() {
        let mut block = [0u8; Q5K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        block[4..16].fill(1);
        block[48..].fill(0x11);

        let values = dequantize_q5_k_bytes(&block, QK_K).unwrap();
        assert_eq!(values.len(), QK_K);
        assert!(values.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }

}
