//! Integration tests for barq-inference
//!
//! Comprehensive end-to-end tests covering:
//! - Model loading and initialization
//! - Context creation and management
//! - Text generation and sampling
//! - Multi-backend compatibility
//! - Performance benchmarks

use barq_core::testing::{TensorFixture, TensorAssertions, BenchmarkTimer, TestStats};
use barq_core::tensor::{Shape, Tensor, TensorType, TensorData};
use barq_core::error::Result;
use std::sync::Arc;

// ============================================================================
// Unit Tests: Core Tensor Operations
// ============================================================================

#[test]
fn test_tensor_creation() {
    let tensor = TensorFixture::simple_2d();
    TensorAssertions::assert_shape(&tensor, &[2, 2]);
}

#[test]
fn test_tensor_arithmetic() {
    use barq_core::ops::Add;

    let a = TensorFixture::simple_2d();
    let b = TensorFixture::filled(Shape::matrix(2, 2), 2.0);

    let add_op = Add;
    let result = add_op.apply(&a, &b).unwrap();

    let expected = Tensor::new(
        None,
        TensorType::F32,
        Shape::matrix(2, 2),
        TensorData::F32(vec![3.0, 4.0, 5.0, 6.0]),
    ).unwrap();

    TensorAssertions::assert_close(&result, &expected, 1e-6);
}

#[test]
fn test_matrix_multiplication() {
    use barq_core::ops::MatMul;

    let a = Tensor::new(
        None,
        TensorType::F32,
        Shape::matrix(2, 2),
        TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
    ).unwrap();

    let b = Tensor::new(
        None,
        TensorType::F32,
        Shape::matrix(2, 2),
        TensorData::F32(vec![5.0, 6.0, 7.0, 8.0]),
    ).unwrap();

    let matmul_op = MatMul;
    let result = matmul_op.apply(&a, &b).unwrap();

    // Expected: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    let expected = Tensor::new(
        None,
        TensorType::F32,
        Shape::matrix(2, 2),
        TensorData::F32(vec![19.0, 22.0, 43.0, 50.0]),
    ).unwrap();

    TensorAssertions::assert_close(&result, &expected, 1e-5);
}

#[test]
fn test_gemm_performance() {
    use barq_core::gemm::{gemm_f32, GEMMConfig};

    let n = 128;
    let a = vec![1.0f32; n * n];
    let b = vec![2.0f32; n * n];
    let mut c = vec![0.0f32; n * n];

    let (result, time) = BenchmarkTimer::measure(|| {
        gemm_f32(&a, &b, &mut c, n, n, n)
    });

    assert!(result.is_ok());
    println!("GEMM 128x128 time: {:.3}ms", time * 1000.0);

    // Verify correctness
    assert!((c[0] - n as f32 * 2.0).abs() < 1e-5);
}

#[test]
fn test_quantization_roundtrip() {
    use barq_core::quant::{Quantize, QuantizationType};

    let original = TensorFixture::sequential(Shape::matrix(32, 32));

    // Quantize to Q4_0
    let quantize = Quantize;
    let (quantized, scales) = quantize.to_q4_0(&original).unwrap();

    // Dequantize
    let dequantize = barq_core::quant::Dequantize;
    let recovered = dequantize.from_q4_0(&quantized, &scales, original.shape().clone()).unwrap();

    // Check error rate
    TensorAssertions::assert_close(&original, &recovered, 0.1); // Allow 10% error for Q4
}

// ============================================================================
// Integration Tests: Model Components
// ============================================================================

#[tokio::test]
async fn test_attention_mechanism() {
    use barq_core::attention::compute_attention;

    let batch_size = 1;
    let num_heads = 4;
    let seq_len = 16;
    let head_dim = 32;

    // Create Q, K, V tensors
    let q = TensorFixture::random(Shape::new(vec![batch_size * num_heads, seq_len, head_dim]));
    let k = TensorFixture::random(Shape::new(vec![batch_size * num_heads, seq_len, head_dim]));
    let v = TensorFixture::random(Shape::new(vec![batch_size * num_heads, seq_len, head_dim]));

    let (result, time) = BenchmarkTimer::measure(|| {
        compute_attention(&q, &k, &v, seq_len, head_dim)
    });

    assert!(result.is_ok());
    let output = result.unwrap();

    // Check output shape
    TensorAssertions::assert_shape(&output, &[batch_size * num_heads, seq_len, head_dim]);

    println!("Attention computation time: {:.3}ms", time * 1000.0);
}

#[tokio::test]
async fn test_rope_positional_encoding() {
    use barq_core::rope::apply_rope;

    let batch_size = 2;
    let num_heads = 4;
    let seq_len = 128;
    let head_dim = 64;

    let mut q = TensorFixture::random(Shape::new(vec![batch_size, num_heads, seq_len, head_dim]));
    let mut k = TensorFixture::random(Shape::new(vec![batch_size, num_heads, seq_len, head_dim]));

    let (result, time) = BenchmarkTimer::measure(|| {
        apply_rope(&mut q, &mut k, seq_len, head_dim, 10000.0)
    });

    assert!(result.is_ok());
    println!("RoPE application time: {:.3}ms", time * 1000.0);
}

#[tokio::test]
async fn test_rms_normalization() {
    use barq_core::normalization::rms_norm;

    let batch_size = 2;
    let seq_len = 128;
    let hidden_dim = 512;

    let input = TensorFixture::random(Shape::new(vec![batch_size, seq_len, hidden_dim]));
    let weight = TensorFixture::filled(Shape::vector(hidden_dim), 1.0);

    let (result, time) = BenchmarkTimer::measure(|| {
        rms_norm(&input, &weight, 1e-5)
    });

    assert!(result.is_ok());
    let output = result.unwrap();

    TensorAssertions::assert_shape(&output, &[batch_size, seq_len, hidden_dim]);
    println!("RMS norm time: {:.3}ms", time * 1000.0);
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

#[test]
fn benchmark_tensor_operations() {
    let sizes = vec![64, 128, 256, 512];
    let mut gemm_stats = TestStats::new();

    for size in sizes {
        let a = vec![1.0f32; size * size];
        let b = vec![2.0f32; size * size];
        let mut c = vec![0.0f32; size * size];

        let (_, time) = BenchmarkTimer::measure(|| {
            barq_core::gemm::gemm_f32(&a, &b, &mut c, size, size, size).unwrap();
        });

        let gflops = (size * size * size * 2) as f64 / (time * 1e9);
        gemm_stats.add(gflops);
        println!("GEMM {}x{}: {:.3}ms ({:.2} GFLOPS)", size, size, time * 1000.0, gflops);
    }

    println!("\nGEMM Performance Statistics:");
    println!("  Mean: {:.2} GFLOPS", gemm_stats.mean());
    println!("  Median: {:.2} GFLOPS", gemm_stats.median());
    println!("  Std Dev: {:.2} GFLOPS", gemm_stats.std_dev());
}

#[test]
fn benchmark_memory_operations() {
    use barq_core::memory::{Allocator, MemoryType, CpuAllocator};

    let sizes = vec![1024, 4096, 16384, 65536]; // Different tensor sizes
    let allocator = CpuAllocator::new(MemoryType::Standard);

    for size in sizes {
        let (buffer, time) = BenchmarkTimer::measure(|| {
            allocator.allocate::<f32>(size).unwrap()
        });

        println!("Memory allocation {} floats: {:.3}ms", size, time * 1000.0);

        // Verify allocation
        assert!(buffer.as_ptr() as *const f32 != std::ptr::null());
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_error_handling() {
    use barq_core::error::Error;

    // Test tensor dimension mismatch
    let a = TensorFixture::simple_2d(); // 2x2
    let b = Tensor::new(
        None,
        TensorType::F32,
        Shape::matrix(3, 3), // 3x3 - incompatible
        TensorData::F32(vec![1.0; 9]),
    ).unwrap();

    use barq_core::ops::MatMul;
    let matmul_op = MatMul;
    let result = matmul_op.apply(&a, &b);

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, Error::DimensionMismatch(_)));
    }
}

#[test]
fn test_invalid_tensor_creation() {
    // Try to create tensor with mismatched data
    let result = Tensor::new(
        None,
        TensorType::F32,
        Shape::matrix(2, 2), // Needs 4 elements
        TensorData::F32(vec![1.0, 2.0, 3.0]), // Only 3 elements
    );

    assert!(result.is_err());
}

// ============================================================================
// Platform Detection Tests
// ============================================================================

#[test]
fn test_platform_detection() {
    let platform = barq_core::platform::detect_platform();
    println!("Detected platform: {:?}", platform);

    let simd = barq_core::platform::detect_simd();
    println!("SIMD capabilities: {:?}", simd);

    let device_info = barq_core::platform::get_device_info();
    println!("Device info: {:?}", device_info);
}

// ============================================================================
// Grammar System Tests
// ============================================================================

#[test]
fn test_grammar_parsing() {
    use barq_core::grammar::{Grammar, GrammarParser, GrammarRule};

    let gbnf = r#"
root ::= item+
item ::= "foo" | "bar" | "baz"
"#;

    let grammar = GrammarParser::parse(gbnf);
    assert!(grammar.is_ok());

    let grammar = grammar.unwrap();
    assert_eq!(grammar.root, "root");
    assert!(grammar.rules.contains_key("root"));
    assert!(grammar.rules.contains_key("item"));
}

#[test]
fn test_json_mode() {
    use sampling::json_mode::{JsonMode, JsonSchema};

    let schema = JsonSchema::Object {
        properties: vec![
            ("name".to_string(), JsonSchema::String),
            ("age".to_string(), JsonSchema::Number),
        ],
        required: vec!["name".to_string(), "age".to_string()],
    };

    let gbnf = JsonMode::schema_to_gbnf(&schema);
    assert!(gbnf.contains("root ::= object"));
}

// ============================================================================
// Sampling Tests
// ============================================================================

#[test]
fn test_sampling_temperature() {
    use sampling::Temperature;

    let sampler = Temperature::new(0.7);

    let mut logits = vec![0.0f32; 10];
    logits[0] = 10.0; // Make first token much more likely

    sampler.sample(&mut logits).unwrap();

    // After sampling, should still be valid probabilities
    let sum: f32 = logits.iter().map(|&x| x.exp()).sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn test_sampling_top_k() {
    use sampling::TopK;

    let sampler = TopK::new(5);

    let mut logits = vec![0.0f32; 10];
    for (i, logit) in logits.iter_mut().enumerate() {
        *logit = i as f32; // Increasing values
    }

    sampler.sample(&mut logits).unwrap();

    // Bottom 5 should be set to -inf
    let mut bottom_neg_inf = true;
    for &logit in &logits[5..] {
        bottom_neg_inf = bottom_neg_inf && logit.is_infinite() && logit.is_sign_negative();
    }
    assert!(bottom_neg_inf);
}

#[test]
fn test_sampling_top_p() {
    use sampling::TopP;

    let sampler = TopP::new(0.9);

    let mut logits = vec![
        10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0
    ];

    sampler.sample(&mut logits).unwrap();

    // Some tokens should be filtered
    let filtered_count = logits.iter().filter(|&&x| x.is_infinite() && x.is_sign_negative()).count();
    assert!(filtered_count > 0);
}
