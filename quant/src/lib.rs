#![allow(
    clippy::all,
    unexpected_cfgs,
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code
)]
#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    non_camel_case_types,
    unused_parens,
    unused_comparisons,
    unreachable_code,
    clippy::needless_update,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::let_and_return,
    clippy::manual_range_contains
)]

//! Quantization algorithms
//!
//! This module provides implementations for various quantization schemes
//! used to compress LLM weights while maintaining accuracy.

pub mod ik_quant;
pub mod iq;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_k;
pub mod q5_k;
pub mod q8_kv;
pub mod simd_quant;
// pub mod blockwise;  // TODO: Implement

pub use ik_quant::{IKQuantConfig, IKQuantType};
pub use q2_k::Q2K;
pub use q3_k::Q3K;
pub use q4_0::Q4_0;
pub use q4_k::Q4_K;
pub use q5_k::Q5K;
pub use q8_kv::Q8KV;
pub use simd_quant::{dequantize_q4_0_simd, matmul_q4_0_simd};
// pub use blockwise::{BlockwiseQuantization, BlockwiseOps};
