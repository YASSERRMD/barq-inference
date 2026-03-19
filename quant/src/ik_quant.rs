//! ik_llama.cpp quantization types
//!
//! This module adds support for the advanced quantization types
//! from ik_llama.cpp that provide better compression and quality.

use std::fs::File;
use std::io::{BufWriter, Write};

use barq_core::error::{Error, Result};
use barq_core::gguf::{GgufReader, GgufValue};

use crate::iq::{quantize_iq, IQQuantConfig, IQType};

/// ik_llama.cpp quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IKQuantType {
    /// IQ4_KS - 4-bit state-of-the-art quantization
    IQ4_KS,
    /// IQ3_KS - 3-bit extreme compression
    IQ3_KS,
    /// IQ2_KS - 2-bit with surprising quality
    IQ2_KS,
    /// Q4_K_R4 - Repacked Q4_K for CPU performance
    Q4_K_R4,
}

impl IKQuantType {
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            IKQuantType::IQ4_KS => 4.0,
            IKQuantType::IQ3_KS => 3.0,
            IKQuantType::IQ2_KS => 2.0,
            IKQuantType::Q4_K_R4 => 4.5,
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            IKQuantType::IQ4_KS => 256,
            IKQuantType::IQ3_KS => 256,
            IKQuantType::IQ2_KS => 256,
            IKQuantType::Q4_K_R4 => 32,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            IKQuantType::IQ4_KS => "IQ4_KS - 4-bit SOTA quantization (recommended for general use)",
            IKQuantType::IQ3_KS => "IQ3_KS - 3-bit extreme compression (edge devices)",
            IKQuantType::IQ2_KS => "IQ2_KS - 2-bit ultra-low VRAM",
            IKQuantType::Q4_K_R4 => "Q4_K_R4 - Repacked for CPU performance",
        }
    }

    pub fn recommended_use(&self) -> &'static str {
        match self {
            IKQuantType::IQ4_KS => "General inference, best quality/size ratio",
            IKQuantType::IQ3_KS => "Memory-constrained edge deployment",
            IKQuantType::IQ2_KS => "Ultra-low VRAM, quality-critical applications",
            IKQuantType::Q4_K_R4 => "CPU-only inference, maximum throughput",
        }
    }
}

/// IK quantization configuration
#[derive(Debug, Clone)]
pub struct IKQuantConfig {
    /// Quantization type
    pub quant_type: IKQuantType,
    /// Enable importance matrix quantization
    pub enable_imatrix: bool,
    /// Number of iterations for importance matrix
    pub imatrix_iterations: usize,
}

impl Default for IKQuantConfig {
    fn default() -> Self {
        Self {
            quant_type: IKQuantType::IQ4_KS,
            enable_imatrix: true,
            imatrix_iterations: 10,
        }
    }
}

impl IKQuantConfig {
    pub fn new(quant_type: IKQuantType) -> Self {
        Self {
            quant_type,
            ..Default::default()
        }
    }

    pub fn cpu_optimized() -> Self {
        Self {
            quant_type: IKQuantType::Q4_K_R4,
            enable_imatrix: false,
            imatrix_iterations: 5,
        }
    }

    pub fn gpu_optimized() -> Self {
        Self {
            quant_type: IKQuantType::IQ4_KS,
            enable_imatrix: true,
            imatrix_iterations: 10,
        }
    }

    pub fn memory_optimized() -> Self {
        Self {
            quant_type: IKQuantType::IQ3_KS,
            enable_imatrix: true,
            imatrix_iterations: 15,
        }
    }

    pub fn ultra_low_memory() -> Self {
        Self {
            quant_type: IKQuantType::IQ2_KS,
            enable_imatrix: true,
            imatrix_iterations: 20,
        }
    }
}

#[derive(Debug, Clone)]
struct QuantizedTensorRecord {
    name: String,
    dims: Vec<u64>,
    original_gguf_type: u32,
    quant_type: IKQuantType,
    data: Vec<u8>,
}

const IKQ_MAGIC: &[u8; 4] = b"IKQ1";
const IKQ_VERSION: u32 = 1;

fn info(msg: &str) {
    println!("{}", msg);
}

fn quant_type_id(qtype: IKQuantType) -> u32 {
    match qtype {
        IKQuantType::IQ4_KS => 1,
        IKQuantType::IQ3_KS => 2,
        IKQuantType::IQ2_KS => 3,
        IKQuantType::Q4_K_R4 => 4,
    }
}

fn to_iq_config(config: &IKQuantConfig) -> IQQuantConfig {
    let iq_type = match config.quant_type {
        IKQuantType::IQ4_KS => IQType::IQ4_KS,
        IKQuantType::IQ3_KS => IQType::IQ3_KS,
        IKQuantType::IQ2_KS => IQType::IQ2_KS,
        IKQuantType::Q4_K_R4 => IQType::Q4_K_R4,
    };

    IQQuantConfig {
        iq_type,
        block_size: match config.quant_type {
            IKQuantType::Q4_K_R4 => 256,
            _ => 256,
        },
    }
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(Error::Io)
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(Error::Io)
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    write_u32(writer, value.len() as u32)?;
    writer.write_all(value.as_bytes()).map_err(Error::Io)
}

fn write_bytes<W: Write>(writer: &mut W, value: &[u8]) -> Result<()> {
    write_u64(writer, value.len() as u64)?;
    writer.write_all(value).map_err(Error::Io)
}

fn write_ikq_archive<W: Write>(
    writer: &mut W,
    config: &IKQuantConfig,
    metadata: &[(String, String)],
    tensors: &[QuantizedTensorRecord],
) -> Result<()> {
    writer.write_all(IKQ_MAGIC).map_err(Error::Io)?;
    write_u32(writer, IKQ_VERSION)?;
    write_u32(writer, quant_type_id(config.quant_type))?;
    write_u32(writer, tensors.len() as u32)?;

    let mut merged_metadata = metadata.to_vec();
    merged_metadata.push((
        "ik.quant_type".to_string(),
        config.quant_type.description().to_string(),
    ));
    merged_metadata.push((
        "ik.quant_bpw".to_string(),
        format!("{:.2}", config.quant_type.bits_per_weight()),
    ));
    merged_metadata.push((
        "ik.enable_imatrix".to_string(),
        config.enable_imatrix.to_string(),
    ));
    merged_metadata.push((
        "ik.imatrix_iterations".to_string(),
        config.imatrix_iterations.to_string(),
    ));

    write_u32(writer, merged_metadata.len() as u32)?;
    for (key, value) in &merged_metadata {
        write_string(writer, key)?;
        write_string(writer, value)?;
    }

    for tensor in tensors {
        write_string(writer, &tensor.name)?;
        write_u32(writer, tensor.dims.len() as u32)?;
        for &dim in &tensor.dims {
            write_u64(writer, dim)?;
        }
        write_u32(writer, tensor.original_gguf_type)?;
        write_u32(writer, quant_type_id(tensor.quant_type))?;
        write_bytes(writer, &tensor.data)?;
    }

    writer.flush().map_err(Error::Io)
}

fn load_quantized_tensors(
    input_model: &str,
    config: &IKQuantConfig,
) -> Result<(Vec<(String, String)>, Vec<QuantizedTensorRecord>)> {
    let mut reader = GgufReader::open(input_model)?;
    let metadata = reader
        .kv_pairs()
        .iter()
        .map(|(key, value)| (key.clone(), format!("{:?}", value)))
        .collect::<Vec<_>>();

    let tensor_infos = reader.tensor_info().to_vec();
    let iq_config = to_iq_config(config);
    let mut tensors = Vec::with_capacity(tensor_infos.len());

    for info in tensor_infos {
        let tensor = reader.load_tensor(&info.name)?;
        let values = tensor.as_f32_slice().map_err(|e| {
            Error::Tensor(format!(
                "Tensor '{}' is not f32-compatible: {}",
                info.name, e
            ))
        })?;

        let data = quantize_iq(values, &iq_config)?;

        tensors.push(QuantizedTensorRecord {
            name: info.name,
            dims: info.dimensions,
            original_gguf_type: info.gguf_type as u32,
            quant_type: config.quant_type,
            data,
        });
    }

    Ok((metadata, tensors))
}

/// Quantize model using IK quantization.
pub fn quantize_model_ik(
    input_model: &str,
    output_model: &str,
    config: &IKQuantConfig,
) -> Result<()> {
    info(&format!("Quantizing model with IK quantization"));
    info(&format!("Input:  {}", input_model));
    info(&format!("Output: {}", output_model));
    info(&format!("Type:   {}", config.quant_type.description()));
    info(&format!(
        "BPW:    {:.2}",
        config.quant_type.bits_per_weight()
    ));
    info(&format!("Block:  {}", config.quant_type.block_size()));

    let (metadata, tensors) = load_quantized_tensors(input_model, config)?;
    let file = File::create(output_model).map_err(Error::Io)?;
    let mut writer = BufWriter::new(file);
    write_ikq_archive(&mut writer, config, &metadata, &tensors)
}

/// Repack existing quantized model for CPU performance.
pub fn repack_model_cpu(input_model: &str, output_model: &str) -> Result<()> {
    info(&format!("Repacking model for CPU performance"));
    info(&format!("Input:  {}", input_model));
    info(&format!("Output: {}", output_model));
    info(&format!("Converting: Q4_K_M → Q4_K_R4"));

    quantize_model_ik(input_model, output_model, &IKQuantConfig::cpu_optimized())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_ik_quant_type_properties() {
        let iq4 = IKQuantType::IQ4_KS;
        assert_eq!(iq4.bits_per_weight(), 4.0);
        assert_eq!(iq4.block_size(), 256);

        let iq3 = IKQuantType::IQ3_KS;
        assert_eq!(iq3.bits_per_weight(), 3.0);

        let iq2 = IKQuantType::IQ2_KS;
        assert_eq!(iq2.bits_per_weight(), 2.0);

        let q4_r4 = IKQuantType::Q4_K_R4;
        assert_eq!(q4_r4.bits_per_weight(), 4.5);
        assert_eq!(q4_r4.block_size(), 32);
    }

    #[test]
    fn test_ik_quant_config_defaults() {
        let config = IKQuantConfig::default();
        assert_eq!(config.quant_type, IKQuantType::IQ4_KS);
        assert!(config.enable_imatrix);
        assert_eq!(config.imatrix_iterations, 10);
    }

    #[test]
    fn test_ik_quant_config_cpu() {
        let config = IKQuantConfig::cpu_optimized();
        assert_eq!(config.quant_type, IKQuantType::Q4_K_R4);
        assert!(!config.enable_imatrix);
    }

    #[test]
    fn test_ik_quant_config_gpu() {
        let config = IKQuantConfig::gpu_optimized();
        assert_eq!(config.quant_type, IKQuantType::IQ4_KS);
        assert!(config.enable_imatrix);
    }

    #[test]
    fn test_ik_quant_config_memory() {
        let config = IKQuantConfig::memory_optimized();
        assert_eq!(config.quant_type, IKQuantType::IQ3_KS);
    }

    #[test]
    fn test_ik_quant_config_ultra_low() {
        let config = IKQuantConfig::ultra_low_memory();
        assert_eq!(config.quant_type, IKQuantType::IQ2_KS);
        assert_eq!(config.imatrix_iterations, 20);
    }

    #[test]
    fn test_archive_writer_header() {
        let config = IKQuantConfig::gpu_optimized();
        let metadata = vec![("general.name".to_string(), "demo".to_string())];
        let tensors = vec![QuantizedTensorRecord {
            name: "weight".to_string(),
            dims: vec![2, 2],
            original_gguf_type: 0,
            quant_type: IKQuantType::IQ4_KS,
            data: vec![1, 2, 3, 4],
        }];

        let mut buffer = Cursor::new(Vec::new());
        write_ikq_archive(&mut buffer, &config, &metadata, &tensors).unwrap();

        let bytes = buffer.into_inner();
        assert_eq!(&bytes[0..4], IKQ_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            IKQ_VERSION
        );
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            quant_type_id(config.quant_type)
        );
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 1);
    }
}
