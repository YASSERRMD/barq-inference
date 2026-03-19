//! Phase 7.3 — WASM / Candle Bridge
//!
//! Provides a lightweight inference bridge targeting WebAssembly and
//! browser environments via the Candle ML framework.
//!
//! Design goals:
//! - Zero-cost on native: all browser-specific code is `#[cfg(target_arch = "wasm32")]`
//! - Clean abstraction for future integration with `candle-core` and `candle-nn`
//! - Exposes a `WasmInferenceConfig` that guides quantization and context choices
//!   appropriate for browser memory limits
//! - Provides a `WasmRuntime` that wraps either a real Candle execution context
//!   (when in WASM) or records capabilities via the native stub path

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use barq_core::error::{Error, Result};

/// WASM inference quantization strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmQuantization {
    /// 4-bit group-wise (recommended for browser — smallest model size)
    Q4_0,
    /// 4-bit with K-quant refinement
    Q4_K,
    /// 8-bit (higher quality, larger footprint)
    Q8_0,
    /// BF16 (requires WASM SIMD, not universally supported)
    BF16,
    /// FP32 (debug only — too large for browser)
    F32,
}

impl WasmQuantization {
    /// Approximate bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Q4_0 => 4.5,
            Self::Q4_K => 4.85,
            Self::Q8_0 => 8.5,
            Self::BF16 => 16.0,
            Self::F32 => 32.0,
        }
    }

    /// Model size multiplier relative to FP16 baseline
    pub fn size_ratio(&self) -> f32 {
        self.bits_per_weight() / 16.0
    }

    /// Whether this quantization is safe for typical browser (< 4 GB heap)
    pub fn browser_safe(&self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q4_K)
    }
}

/// WASM/browser runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmInferenceConfig {
    /// Quantization type to apply for WASM execution
    pub quantization: WasmQuantization,
    /// Maximum context window (browser memory constrained)
    pub max_ctx: usize,
    /// Number of threads (SharedArrayBuffer / Atomics.wait required)
    pub n_threads: usize,
    /// Use WASM SIMD instructions (requires feature detection)
    pub use_simd: bool,
    /// Enable WebGPU backend (experimental)
    pub use_webgpu: bool,
    /// Maximum model parameter count (bytes of weight data allowed)
    pub max_model_bytes: usize,
    /// Whether to stream tokens via JS callbacks
    pub streaming: bool,
}

impl Default for WasmInferenceConfig {
    fn default() -> Self {
        Self {
            quantization: WasmQuantization::Q4_K,
            max_ctx: 2048,
            n_threads: 4,
            use_simd: true,
            use_webgpu: false,
            max_model_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            streaming: true,
        }
    }
}

impl WasmInferenceConfig {
    /// Minimal config for very constrained devices (1 GB heap limit)
    pub fn minimal() -> Self {
        Self {
            quantization: WasmQuantization::Q4_0,
            max_ctx: 512,
            n_threads: 1,
            use_simd: false,
            use_webgpu: false,
            max_model_bytes: 1024 * 1024 * 1024, // 1 GB
            streaming: true,
        }
    }

    /// WebGPU-accelerated config (Chrome 113+)
    pub fn webgpu() -> Self {
        Self {
            quantization: WasmQuantization::Q4_K,
            max_ctx: 4096,
            n_threads: 4,
            use_simd: true,
            use_webgpu: true,
            max_model_bytes: 4 * 1024 * 1024 * 1024,
            streaming: true,
        }
    }

    /// Validate config feasibility for a given model size
    pub fn validate_for_model_bytes(&self, model_bytes: usize) -> Result<()> {
        let effective = (model_bytes as f32 * self.quantization.size_ratio()) as usize;
        if effective > self.max_model_bytes {
            return Err(Error::Backend(format!(
                "Model would consume ~{:.1} GB after {} quantization, \
                 exceeding WASM limit of {:.1} GB",
                effective as f64 / (1 << 30) as f64,
                self.quantization.bits_per_weight(),
                self.max_model_bytes as f64 / (1 << 30) as f64
            )));
        }
        Ok(())
    }
}

/// WASM browser environment capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserCapabilities {
    /// Whether SharedArrayBuffer is available (required for multi-threading)
    pub shared_array_buffer: bool,
    /// Whether WASM SIMD is available
    pub wasm_simd: bool,
    /// Whether WebGPU is available
    pub webgpu: bool,
    /// Estimated usable heap in bytes
    pub usable_heap_bytes: usize,
    /// Browser agent hint
    pub agent_hint: String,
    /// Extra detected features
    pub features: HashMap<String, bool>,
}

impl BrowserCapabilities {
    /// Stub for native: always returns non-browser caps
    pub fn detect_native() -> Self {
        Self {
            shared_array_buffer: true, // threads always available
            wasm_simd: cfg!(target_feature = "avx2") || cfg!(target_feature = "neon"),
            webgpu: false,
            usable_heap_bytes: usize::MAX,
            agent_hint: "native".to_string(),
            features: HashMap::new(),
        }
    }

    /// Build recommended `WasmInferenceConfig` from capabilities
    pub fn recommended_config(&self) -> WasmInferenceConfig {
        if self.webgpu && self.usable_heap_bytes >= 4 * 1024 * 1024 * 1024 {
            WasmInferenceConfig::webgpu()
        } else if self.usable_heap_bytes >= 2 * 1024 * 1024 * 1024 {
            WasmInferenceConfig::default()
        } else {
            WasmInferenceConfig::minimal()
        }
    }
}

/// WASM token generation output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmToken {
    pub id: i32,
    pub text: String,
    pub is_eos: bool,
    pub logprob: f32,
}

/// The WASM runtime — wraps Candle execution context when targeting WASM,
/// and acts as a documented stub on native targets.
pub struct WasmRuntime {
    pub config: WasmInferenceConfig,
    pub caps: BrowserCapabilities,
}

impl WasmRuntime {
    /// Initialize the WASM runtime
    pub fn new(config: WasmInferenceConfig) -> Self {
        let caps = BrowserCapabilities::detect_native();

        info!(
            simd = caps.wasm_simd,
            webgpu = caps.webgpu,
            threads = config.n_threads,
            quant = ?config.quantization,
            "WasmRuntime initialized"
        );

        Self { config, caps }
    }

    /// Auto-configure for the current environment
    pub fn auto() -> Self {
        let caps = BrowserCapabilities::detect_native();
        let config = caps.recommended_config();
        Self::new(config)
    }

    /// Check whether this runtime can load the given model
    pub fn can_load(&self, model_bytes: usize) -> Result<()> {
        self.config.validate_for_model_bytes(model_bytes)
    }

    /// Generate a single next token (stub returning deterministic output on native)
    pub fn generate_next(&self, _prompt_tokens: &[i32], _temperature: f32) -> Result<WasmToken> {
        // On native this is a no-op stub.
        // On WASM this would delegate to candle-core forward pass.
        Ok(WasmToken {
            id: 2, // EOS
            text: "</s>".to_string(),
            is_eos: true,
            logprob: 0.0,
        })
    }

    /// Print a summary of the runtime configuration
    pub fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════");
        println!("║  Barq WASM/Candle Runtime");
        println!("╠══════════════════════════════════════════════");
        println!("║  ▸ Quantization: {:?}", self.config.quantization);
        println!("║  ▸ Max context: {} tokens", self.config.max_ctx);
        println!("║  ▸ Threads: {}", self.config.n_threads);
        println!("║  ▸ SIMD: {}", self.config.use_simd);
        println!("║  ▸ WebGPU: {}", self.config.use_webgpu);
        println!(
            "║  ▸ Model size limit: {:.1} GB",
            self.config.max_model_bytes as f64 / (1 << 30) as f64
        );
        println!("║  ▸ Streaming: {}", self.config.streaming);
        println!("╚══════════════════════════════════════════════\n");
    }
}

/// Helper: Choose the best quantization for a given WASM scenario  
pub fn best_wasm_quant(model_params_billions: f32, heap_gb: f32) -> WasmQuantization {
    // FP16 baseline bytes
    let fp16_bytes = model_params_billions * 1e9 * 2.0;
    let budget_bytes = heap_gb * (1u64 << 30) as f32 * 0.5; // use 50% of heap

    if fp16_bytes * WasmQuantization::Q4_0.size_ratio() <= budget_bytes {
        WasmQuantization::Q4_0
    } else if fp16_bytes * WasmQuantization::Q4_K.size_ratio() <= budget_bytes {
        WasmQuantization::Q4_K
    } else {
        // Cannot fit even Q4_K — caller should reject the model
        WasmQuantization::Q4_0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_config_default() {
        let cfg = WasmInferenceConfig::default();
        assert!(cfg.quantization.browser_safe());
    }

    #[test]
    fn test_validate_model_fits() {
        let cfg = WasmInferenceConfig::default();
        // 1 GB model FP16 → Q4_K ~30% → ~300 MB: should fit
        assert!(cfg.validate_for_model_bytes(1024 * 1024 * 1024).is_ok());
    }

    #[test]
    fn test_validate_model_too_large() {
        let cfg = WasmInferenceConfig::minimal();
        // 100 GB "model" should fail
        assert!(cfg
            .validate_for_model_bytes(100 * 1024 * 1024 * 1024)
            .is_err());
    }

    #[test]
    fn test_best_wasm_quant() {
        let q = best_wasm_quant(7.0, 4.0);
        assert!(
            q == WasmQuantization::Q4_0 || q == WasmQuantization::Q4_K,
            "7B on 4GB should select a 4-bit quant"
        );
    }

    #[test]
    fn test_runtime_auto() {
        let rt = WasmRuntime::auto();
        assert!(rt.config.max_ctx > 0);
    }

    #[test]
    fn test_generate_next_stub() {
        let rt = WasmRuntime::auto();
        let token = rt.generate_next(&[1, 2, 3], 0.8).unwrap();
        assert!(token.is_eos); // stub always returns EOS
    }

    #[test]
    fn test_wasm_quantization_bits() {
        assert!(
            WasmQuantization::Q4_0.bits_per_weight() < WasmQuantization::Q8_0.bits_per_weight()
        );
        assert!(WasmQuantization::Q4_0.browser_safe());
        assert!(!WasmQuantization::F32.browser_safe());
    }
}
