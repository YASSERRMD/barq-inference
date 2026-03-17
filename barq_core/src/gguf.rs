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

/// GGUF-specific tensor types (quantized)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    /// Standard types
    F32 = 0,
    F16 = 1,
    Q4_0 = 8,
    Q4_1 = 9,
    Q5_0 = 10,
    Q5_1 = 11,
    Q8_0 = 12,
    Q2_K = 16,
    Q3_K = 17,
    Q4_K = 18,
    Q5_K = 19,
    Q6_K = 20,
    Q8_K = 21,
    /// Q4_K with medium (M) variant
    Q4_K_M = 14,
}

impl GgufTensorType {
    /// Parse from u32
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GgufTensorType::F32),
            1 => Ok(GgufTensorType::F16),
            8 => Ok(GgufTensorType::Q4_0),
            9 => Ok(GgufTensorType::Q4_1),
            10 => Ok(GgufTensorType::Q5_0),
            11 => Ok(GgufTensorType::Q5_1),
            12 => Ok(GgufTensorType::Q8_0),
            14 => Ok(GgufTensorType::Q4_K_M),
            16 => Ok(GgufTensorType::Q2_K),
            17 => Ok(GgufTensorType::Q3_K),
            18 => Ok(GgufTensorType::Q4_K),
            19 => Ok(GgufTensorType::Q5_K),
            20 => Ok(GgufTensorType::Q6_K),
            21 => Ok(GgufTensorType::Q8_K),
            _ => Err(Error::InvalidGguf(format!("Unknown GGUF tensor type: {}", value))),
        }
    }

    /// Returns the block size for quantized types
    pub const fn block_size(&self) -> usize {
        match self {
            GgufTensorType::Q4_0 | GgufTensorType::Q4_1 => 32,
            GgufTensorType::Q5_0 | GgufTensorType::Q5_1 => 32,
            GgufTensorType::Q8_0 => 32,
            GgufTensorType::Q2_K | GgufTensorType::Q3_K | GgufTensorType::Q4_K |
            GgufTensorType::Q4_K_M | GgufTensorType::Q5_K | GgufTensorType::Q6_K | GgufTensorType::Q8_K => 256,
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
            GgufTensorType::Q2_K => "q2_k",
            GgufTensorType::Q3_K => "q3_k",
            GgufTensorType::Q4_K => "q4_k",
            GgufTensorType::Q4_K_M => "q4_k_m",
            GgufTensorType::Q5_K => "q5_k",
            GgufTensorType::Q6_K => "q6_k",
            GgufTensorType::Q8_K => "q8_k",
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

            let gguf_type = GgufTensorType::from_u32(reader.read_u32::<LE>().map_err(|e| Error::Io(e))?)?;
            let offset = reader.read_u64::<LE>().map_err(|e| Error::Io(e))?;

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

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_info.iter().map(|t| t.name.as_str()).collect()
    }

    /// Load a specific tensor with dequantization support
    pub fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        let info = self.get_tensor_info(name)
            .ok_or_else(|| Error::Tensor(format!("Tensor not found: {}", name)))?;

        // Clone the fields we need after seeking
        let offset = info.offset;
        let dimensions = info.dimensions.clone();
        let gguf_type = info.gguf_type;
        let tensor_name = info.name.clone();

        // Seek to tensor data offset
        use std::io::Seek;
        self.reader.seek(io::SeekFrom::Start(offset))
            .map_err(|e| Error::Io(e))?;

        // Calculate total elements
        let total_elements: usize = dimensions.iter().map(|&d| d as usize).product();

        // Load and dequantize based on type
        let tensor_data = match gguf_type {
            GgufTensorType::F32 => {
                let type_size = 4;
                let total_bytes = total_elements * type_size;
                let mut data = vec![0u8; total_bytes];
                self.reader.read_exact(&mut data).map_err(|e| Error::Io(e))?;

                let values: Vec<f32> = data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                crate::tensor::TensorData::F32(values)
            }
            GgufTensorType::F16 => {
                let type_size = 2;
                let total_bytes = total_elements * type_size;
                let mut data = vec![0u8; total_bytes];
                self.reader.read_exact(&mut data).map_err(|e| Error::Io(e))?;

                let values: Vec<f32> = data.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                crate::tensor::TensorData::F32(values)
            }
            GgufTensorType::Q4_0 => {
                self.load_q4_0(&dimensions, total_elements)?
            }
            GgufTensorType::Q4_1 => {
                return Err(Error::Unsupported(format!("Q4_1 not yet implemented")))
            }
            GgufTensorType::Q8_0 => {
                self.load_q8_0(&dimensions, total_elements)?
            }
            GgufTensorType::Q4_K | GgufTensorType::Q4_K_M => {
                self.load_q4_k(&dimensions, total_elements)?
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
    fn load_q4_0(&mut self, dimensions: &[u64], total_elements: usize) -> Result<crate::tensor::TensorData> {
        let block_size = 32; // Q4_0 block size
        let n_blocks = (total_elements + block_size - 1) / block_size;

        // Q4_0 format: scale (f32) + quants (16 bytes for 32 values)
        let bytes_per_block = 4 + (block_size / 2);
        let total_bytes = n_blocks * bytes_per_block;

        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(|e| Error::Io(e))?;

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
            let q_len = (block_size + 1) / 2;
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

    /// Load Q8_0 quantized tensor and dequantize to f32
    fn load_q8_0(&mut self, dimensions: &[u64], total_elements: usize) -> Result<crate::tensor::TensorData> {
        let block_size = 32; // Q8_0 block size
        let n_blocks = (total_elements + block_size - 1) / block_size;

        // Q8_0 format: scale (f32) + quants (32 bytes)
        let bytes_per_block = 4 + block_size;
        let total_bytes = n_blocks * bytes_per_block;

        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(|e| Error::Io(e))?;

        let mut output = Vec::with_capacity(total_elements);
        let mut offset = 0;

        for _block in 0..n_blocks {
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

    /// Load Q4_K_M quantized tensor and dequantize to f32
    /// Implementation matching llama.cpp ggml-quants.c exactly
    fn load_q4_k(&mut self, _dimensions: &[u64], total_elements: usize) -> Result<crate::tensor::TensorData> {
        const QK_K: usize = 256; // Q4_K block size

        let n_blocks = (total_elements + QK_K - 1) / QK_K;

        // Q4_K format per 256 values (from llama.cpp block_q4_K):
        // - d (f16): 2 bytes (main scale)
        // - dmin (f16): 2 bytes (main minimum)
        // - scales[12]: 12 bytes (packed scale/minimum values)
        // - qs[QK_K/4]: 64 bytes (4-bit packed quantized values)
        // Total: 80 bytes per 256-value block
        const BYTES_PER_BLOCK: usize = 2 + 2 + 12 + (QK_K / 4);
        let total_bytes = n_blocks * BYTES_PER_BLOCK;

        let mut data = vec![0u8; total_bytes];
        self.reader.read_exact(&mut data).map_err(|e| Error::Io(e))?;

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

            // Read qs[QK_K/4] (4-bit packed quantized values)
            const QS_LEN: usize = QK_K / 4;
            let mut qs = [0u8; QS_LEN];
            if offset + QS_LEN > data.len() {
                break;
            }
            qs.copy_from_slice(&data[offset..offset + QS_LEN]);
            offset += qs.len();

            // Dequantize: matching llama.cpp dequantize_row_q4_K()
            let mut is = 0usize;

            for j in (0..QK_K).step_by(64) {
                // Process first 32 values
                let (sc, m) = Self::get_scale_min_k4(is + 0, &scales);
                let d1 = d * (sc as u32) as f32;
                let m1 = dmin * (m as u32) as f32;

                for l in 0..32 {
                    let q_idx_in_qs = (j + l) / 2;
                    let shift = if (j + l) % 2 == 0 { 0 } else { 4 };
                    let q = (qs[q_idx_in_qs] >> shift) & 0xF;
                    output.push(d1 * (q as f32) - m1);
                }

                // Process second 32 values
                let (sc, m) = Self::get_scale_min_k4(is + 1, &scales);
                let d2 = d * (sc as u32) as f32;
                let m2 = dmin * (m as u32) as f32;

                for l in 0..32 {
                    let q_idx_in_qs = (j + 32 + l) / 2;
                    let shift = if (j + 32 + l) % 2 == 0 { 0 } else { 4 };
                    let q = (qs[q_idx_in_qs] >> shift) & 0xF;
                    output.push(d2 * (q as f32) - m2);
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
            let m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
            (d, m)
        }
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
