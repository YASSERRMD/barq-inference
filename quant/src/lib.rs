//! Quantization algorithms
//!
//! This module provides implementations for various quantization schemes
//! used to compress LLM weights while maintaining accuracy.

pub mod iq;
pub mod q4_0;
pub mod q4_k;
pub mod simd_quant;
// pub mod ik_quant;  // TODO: Implement
// pub mod blockwise;  // TODO: Implement

pub use q4_0::Q4_0;
pub use q4_k::Q4_K;
pub use simd_quant::{dequantize_q4_0_simd, matmul_q4_0_simd};
// pub use ik_quant::{IKQuantType, IKQuantConfig, quantize_model_ik, repack_model_cpu};
// pub use blockwise::{BlockwiseQuantization, BlockwiseOps};
