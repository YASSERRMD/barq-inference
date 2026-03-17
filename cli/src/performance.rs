//! Performance optimization helpers
//!
//! Provides utilities for enabling performance optimizations like CUDA Graphs

/// Enable CUDA Graphs optimization for NVIDIA GPUs
///
/// CUDA Graphs eliminate per-step CPU→GPU kernel launch overhead
/// by recording and replaying the compute graph. Expected gain: 7-20% TPS.
///
/// # Example
/// ```ignore
/// use cli::performance::enable_cuda_graphs;
///
/// enable_cuda_graphs(true);  // Enable CUDA Graphs
/// ```
pub fn enable_cuda_graphs(enable: bool) {
    if enable {
        std::env::set_var("GGML_CUDA_GRAPH_OPT", "1");
    } else {
        std::env::remove_var("GGML_CUDA_GRAPH_OPT");
    }
}

/// Check if CUDA Graphs is enabled
pub fn cuda_graphs_enabled() -> bool {
    std::env::var("GGML_CUDA_GRAPH_OPT")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Enable Flash Attention via environment variable
///
/// This provides an alternative way to enable Flash Attention
/// in addition to the ContextParams flag.
pub fn enable_flash_attention(enable: bool) {
    if enable {
        std::env::set_var("GGML_FLASH_ATTN", "1");
    } else {
        std::env::remove_var("GGML_FLASH_ATTN");
    }
}

/// Check if Flash Attention is enabled
pub fn flash_attention_enabled() -> bool {
    std::env::var("GGML_FLASH_ATTN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Performance optimization presets
#[derive(Debug, Clone, Copy)]
pub enum PerformancePreset {
    /// Maximum speed (uses all optimizations)
    MaxSpeed,
    /// Balanced speed and quality
    Balanced,
    /// Maximum quality (slower but better)
    MaxQuality,
    /// CPU inference optimizations
    CPU,
    /// GPU inference optimizations
    GPU,
}

impl PerformancePreset {
    /// Apply performance preset via environment variables
    pub fn apply(self) {
        match self {
            PerformancePreset::MaxSpeed => {
                enable_cuda_graphs(true);
                enable_flash_attention(true);
                std::env::set_var("GGML_CUDA_GRAPHS", "1");
                std::env::set_var("GGML_CUDA_FA_ALL_QUANTS", "1");
            }
            PerformancePreset::Balanced => {
                enable_cuda_graphs(true);
                enable_flash_attention(true);
            }
            PerformancePreset::MaxQuality => {
                // Quality mode: enable Flash Attention but not CUDA Graphs
                // (Graphs can sometimes have precision issues)
                enable_cuda_graphs(false);
                enable_flash_attention(true);
            }
            PerformancePreset::CPU => {
                // CPU optimizations
                enable_cuda_graphs(false);
                enable_flash_attention(false);
                std::env::set_var("GGML_NUM_THREADS", num_cpus::get_physical().to_string());
            }
            PerformancePreset::GPU => {
                enable_cuda_graphs(true);
                enable_flash_attention(true);
                std::env::set_var("GGML_CUDA_GRAPHS", "1");
                std::env::set_var("GGML_CUDA_FA_ALL_QUANTS", "1");
                std::env::set_var("GGML_NUM_THREADS", "4"); // CPU is bottleneck
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_all_presets() {
        // Note: These tests manipulate global environment variables and must run sequentially.
        // Combining them into one test function ensures they don't race.

        // Test basic toggles
        enable_cuda_graphs(true);
        assert!(cuda_graphs_enabled());
        enable_cuda_graphs(false);
        assert!(!cuda_graphs_enabled());

        enable_flash_attention(true);
        assert!(flash_attention_enabled());
        enable_flash_attention(false);
        assert!(!flash_attention_enabled());

        // Test MaxSpeed preset
        PerformancePreset::MaxSpeed.apply();
        assert!(cuda_graphs_enabled());
        assert!(flash_attention_enabled());

        // Test CPU preset
        PerformancePreset::CPU.apply();
        assert!(!cuda_graphs_enabled());
        assert!(!flash_attention_enabled());

        // Test MaxQuality preset
        PerformancePreset::MaxQuality.apply();
        assert!(!cuda_graphs_enabled());
        assert!(flash_attention_enabled());
    }
}
