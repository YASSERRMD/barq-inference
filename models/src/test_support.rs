//! Test support helpers for model fixtures.
//!
//! These helpers generate tiny GGUF files with metadata only so unit tests can
//! exercise the real loader and architecture detection code without needing
//! full model weights.

use barq_core::gguf::{GgufType, GgufValue, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Write a minimal GGUF file containing only metadata key/value pairs.
///
/// The file has zero tensors and is aligned to the default GGUF alignment so
/// it looks like a normal model header to the loader.
pub(crate) fn write_test_gguf_file(prefix: &str, kv_pairs: &[(&str, GgufValue)]) -> PathBuf {
    let path = unique_test_path(prefix);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("failed to create temporary GGUF directory");
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(GGUF_MAGIC);
    bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());

    for (key, value) in kv_pairs {
        write_string(&mut bytes, key);
        bytes.extend_from_slice(&(value.get_type() as u32).to_le_bytes());
        write_value(&mut bytes, value);
    }

    let alignment = DEFAULT_ALIGNMENT as usize;
    let padding = (alignment - (bytes.len() % alignment)) % alignment;
    bytes.extend(std::iter::repeat_n(0u8, padding));

    let mut file = File::create(&path).expect("failed to create temporary GGUF file");
    file.write_all(&bytes)
        .expect("failed to write temporary GGUF file");

    path
}

fn unique_test_path(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX_EPOCH")
        .as_nanos();
    let pid = std::process::id();
    std::env::temp_dir()
        .join(format!("barq-{}-{}-{}", prefix, pid, nanos))
        .join("fixture.gguf")
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn write_value(bytes: &mut Vec<u8>, value: &GgufValue) {
    match value {
        GgufValue::Uint8(v) => bytes.push(*v),
        GgufValue::Int8(v) => bytes.push(*v as u8),
        GgufValue::Uint16(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Int16(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Uint32(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Int32(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Uint64(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Int64(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Float32(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Float64(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Bool(v) => bytes.push(u8::from(*v)),
        GgufValue::String(value) => write_string(bytes, value),
        GgufValue::Array(values) => {
            let element_type = values
                .first()
                .map(GgufValue::get_type)
                .unwrap_or(GgufType::Uint8);
            bytes.extend_from_slice(&(element_type as u32).to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for item in values {
                write_value(bytes, item);
            }
        }
    }
}
