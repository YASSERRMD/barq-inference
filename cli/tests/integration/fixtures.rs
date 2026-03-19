use barq_core::gguf::{GgufType, GgufValue, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) struct TensorSpec {
    pub name: &'static str,
    pub dimensions: Vec<u64>,
    pub values: Vec<f32>,
}

pub(crate) fn tiny_llama_fixture(prefix: &str) -> PathBuf {
    let tokens = tiny_llama_tokens();
    let vocab_size = tokens.len();
    let hidden = 4usize;

    let mut token_embd = vec![0.0f32; vocab_size * hidden];
    token_embd[2 * hidden] = 1.0;

    let mut output = vec![0.0f32; vocab_size * hidden];
    output[3 * hidden] = 10.0;

    let metadata = vec![
        (
            "general.architecture",
            GgufValue::String("llama".to_string()),
        ),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
        ),
        ("llama.block_count", GgufValue::Uint32(0)),
        ("llama.attention.head_count", GgufValue::Uint32(1)),
        ("llama.attention.head_count_kv", GgufValue::Uint32(1)),
        ("llama.embedding_length", GgufValue::Uint32(hidden as u32)),
        (
            "llama.feed_forward_length",
            GgufValue::Uint32(hidden as u32),
        ),
        ("llama.context_length", GgufValue::Uint32(16)),
        (
            "llama.vocabulary_size",
            GgufValue::Uint32(vocab_size as u32),
        ),
        ("llama.rope.freq_base", GgufValue::Float32(10000.0)),
        ("llama.rope.freq_scale", GgufValue::Float32(1.0)),
        ("llama.rope.scaling.type", GgufValue::Uint32(0)),
        ("general.alignment", GgufValue::Uint32(DEFAULT_ALIGNMENT)),
    ];

    let tensors = vec![
        TensorSpec {
            name: "token_embd.weight",
            dimensions: vec![vocab_size as u64, hidden as u64],
            values: token_embd,
        },
        TensorSpec {
            name: "output.weight",
            dimensions: vec![vocab_size as u64, hidden as u64],
            values: output,
        },
    ];

    write_gguf_fixture(prefix, &metadata, &tensors)
}

fn tiny_llama_tokens() -> Vec<String> {
    let mut tokens = Vec::with_capacity(128);
    tokens.push("<s>".to_string());
    tokens.push("</s>".to_string());
    tokens.push("▁hello".to_string());
    tokens.push("world".to_string());

    for idx in 4..128 {
        tokens.push(format!("tok_{}", idx));
    }

    tokens
}

fn write_gguf_fixture(
    prefix: &str,
    kv_pairs: &[(&str, GgufValue)],
    tensors: &[TensorSpec],
) -> PathBuf {
    let path = unique_test_path(prefix);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("failed to create temporary GGUF directory");
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(GGUF_MAGIC);
    bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());

    for (key, value) in kv_pairs {
        write_string(&mut bytes, key);
        bytes.extend_from_slice(&(value.get_type() as u32).to_le_bytes());
        write_value(&mut bytes, value);
    }

    let alignment = DEFAULT_ALIGNMENT as usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut current_offset = 0usize;
    for tensor in tensors {
        current_offset = align_to(current_offset, alignment);
        offsets.push(current_offset as u64);
        current_offset += tensor.values.len() * std::mem::size_of::<f32>();
    }

    for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
        write_string(&mut bytes, tensor.name);
        bytes.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dim in &tensor.dimensions {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&(0u32).to_le_bytes());
        bytes.extend_from_slice(&(*offset).to_le_bytes());
    }

    let padding = (alignment - (bytes.len() % alignment)) % alignment;
    bytes.extend(std::iter::repeat_n(0u8, padding));

    let mut data_bytes = Vec::new();
    for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
        let target_len = *offset as usize;
        if data_bytes.len() < target_len {
            data_bytes.extend(std::iter::repeat_n(0u8, target_len - data_bytes.len()));
        }
        for value in &tensor.values {
            data_bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    bytes.extend_from_slice(&data_bytes);

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

fn align_to(value: usize, alignment: usize) -> usize {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
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
