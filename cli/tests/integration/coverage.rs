use barq_core::{
    attention::{AttentionConfig, MultiHeadAttention},
    rope,
    testing::{TensorAssertions, TensorFixture},
};
use models::{arch_registry::ArchitectureRegistry, LlmArch};
use quant::{Q2K, Q3K, Q5K, Q8KV};
use sampling::{
    sampler::{Sampler, TokenData},
    JsonMode, JsonSchema, SamplerChain, Temperature, TopK, TopP,
};

#[test]
fn quantization_roundtrip_smoke() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 50.0).collect();

    let q2 = Q2K::new();
    let q2_bytes = q2.quantize(&input).expect("q2 quantize");
    let q2_out = q2
        .dequantize(&q2_bytes, input.len())
        .expect("q2 dequantize");
    assert_eq!(q2_out.len(), input.len());

    let q3 = Q3K::new();
    let q3_bytes = q3.quantize(&input).expect("q3 quantize");
    let q3_out = q3
        .dequantize(&q3_bytes, input.len())
        .expect("q3 dequantize");
    assert_eq!(q3_out.len(), input.len());

    let q5 = Q5K::new();
    let q5_bytes = q5.quantize(&input).expect("q5 quantize");
    let q5_out = q5
        .dequantize(&q5_bytes, input.len())
        .expect("q5 dequantize");
    assert_eq!(q5_out.len(), input.len());

    let q8 = Q8KV::new();
    let q8_bytes = q8.quantize(&input).expect("q8 quantize");
    let q8_out = q8
        .dequantize(&q8_bytes, input.len())
        .expect("q8 dequantize");
    assert_eq!(q8_out.len(), input.len());
}

#[test]
fn architecture_registry_smoke() {
    assert_eq!(
        ArchitectureRegistry::from_name("llama"),
        Some(LlmArch::Llama)
    );
    assert_eq!(
        ArchitectureRegistry::from_name("qwen2.moe"),
        Some(LlmArch::Qwen2Moe)
    );
    assert!(ArchitectureRegistry::is_supported(LlmArch::DeepSeekMoE));
    assert!(ArchitectureRegistry::supported_architectures().contains(&LlmArch::Mixtral));
}

#[test]
fn sampling_pipeline_smoke() {
    let mut chain = SamplerChain::new()
        .add(Box::new(TopK::new(1)))
        .add(Box::new(Temperature::new(0.0)));
    let mut logits = vec![
        TokenData::new(0, 1.0),
        TokenData::new(1, 3.0),
        TokenData::new(2, 2.0),
    ];

    let token = chain.sample(&mut logits).expect("chain sample");
    assert_eq!(token, 1);

    let mut top_p = TopP::new(1.0);
    let mut logits = vec![TokenData::new(0, 0.5), TokenData::new(1, 1.5)];
    assert_eq!(top_p.sample(&mut logits).expect("top-p sample"), 1);

    let schema = JsonSchema::Object {
        properties: vec![],
        required: vec![],
    };
    let sampler = JsonMode::create_sampler(&schema, 128).expect("json sampler");
    assert!(!sampler.get_allowed_tokens().is_empty());
}

#[test]
fn attention_and_rope_smoke() {
    let config = AttentionConfig::new(4, 8, true);
    let attention = MultiHeadAttention::new(config);

    let q = vec![1.0f32; 4 * 8];
    let k = vec![1.0f32; 4 * 8];
    let v = vec![1.0f32; 4 * 8];
    let output = attention
        .scaled_dot_product_attention(&q, &k, &v)
        .expect("attention should succeed");
    assert_eq!(output.len(), 4 * 8);

    let positions = vec![0, 1];
    let (cos, sin) = rope::rope(&positions, 8, 10000.0, 1.0).expect("rope should succeed");
    let mut q = vec![1.0f32; 16];
    let mut k = vec![2.0f32; 16];
    rope::apply_rope(&mut q, &mut k, &cos, &sin, 8).expect("rope application");
    assert!(q.iter().any(|&value| (value - 1.0).abs() > f32::EPSILON));

    let tensor = TensorFixture::simple_2d();
    TensorAssertions::assert_shape(&tensor, &[2, 2]);
}
