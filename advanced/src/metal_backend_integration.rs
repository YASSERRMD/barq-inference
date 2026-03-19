//! Phase 7.2 — Metal Backend Integration
//!
//! Bridges the runtime Metal detection (`metal_detect`) with the
//! inference engine's context parameters, providing auto-tuned defaults
//! for Apple Silicon workloads.
//!
//! Key responsibilities:
//! - Auto-select GPU vs CPU backend based on Metal capability report
//! - Tune `ContextParams` for the detected chip (threads, layers, ctx size)
//! - Provide Metal-aware performance presets
//! - Expose a single `apply_metal_optimizations()` function usable from the CLI

use tracing::info;

use crate::metal_detect::{detect, AppleChipGeneration, MetalCapabilities};
use models::context::ContextParams;

/// Metal-tuned inference preset
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalPreset {
    /// Maximum throughput (offload all layers, full context)
    MaxThroughput,
    /// Balanced (offload most layers, moderate context)
    Balanced,
    /// Memory-saving (fewer GPU layers, smaller context)
    MemorySaver,
    /// CPU-only fallback (no GPU)
    CpuOnly,
}

/// Result of applying Metal optimizations
#[derive(Debug, Clone)]
pub struct MetalOptimizationReport {
    pub caps: MetalCapabilities,
    pub preset: MetalPreset,
    pub params: ContextParams,
    pub gpu_layers: u32,
    pub notes: Vec<String>,
}

/// Auto-detect Metal capabilities and return optimized `ContextParams`.
///
/// This is the main entry point for Phase 7.2.
pub fn apply_metal_optimizations(base: ContextParams) -> MetalOptimizationReport {
    let caps = detect();
    tune_for_capabilities(caps, base)
}

/// Given a `MetalCapabilities` report, produce the best `ContextParams`.
pub fn tune_for_capabilities(
    caps: MetalCapabilities,
    base: ContextParams,
) -> MetalOptimizationReport {
    let mut params = base;
    let mut notes = Vec::new();

    if !caps.metal_available {
        // Pure CPU path
        notes.push("Metal not available — using CPU-only mode".to_string());
        params.flash_attn = false;
        params.n_gpu_layers = 0;
        params.n_threads = num_cpus::get() as u32;
        return MetalOptimizationReport {
            caps,
            preset: MetalPreset::CpuOnly,
            params,
            gpu_layers: 0,
            notes,
        };
    }

    // Metal is available — pick preset based on chip and memory
    let gb = caps.unified_memory_bytes / (1 << 30);
    let preset = choose_preset(&caps, gb);

    let gpu_layers = match &preset {
        MetalPreset::MaxThroughput => 9999, // all layers
        MetalPreset::Balanced => caps.chip_generation.recommended_gpu_layers_7b(),
        MetalPreset::MemorySaver => caps.chip_generation.recommended_gpu_layers_7b() / 2,
        MetalPreset::CpuOnly => 0,
    };

    // Context window — scale with unified memory
    let n_ctx = match gb {
        0..=7 => 4096,
        8..=15 => 8192,
        16..=31 => 16384,
        32..=63 => 32768,
        _ => 65536,
    }
    .min(caps.max_recommended_ctx as u32);

    // Threads — Apple Silicon has efficient cores, fewer is better for throughput
    let n_threads = if caps.is_apple_silicon {
        4
    } else {
        num_cpus::get() as u32
    };

    // Apply to params
    params.n_gpu_layers = gpu_layers as i32;
    params.n_ctx = n_ctx;
    params.n_threads = n_threads;
    params.flash_attn = caps.is_apple_silicon; // Flash Attention pays off on MPS

    notes.push(format!("Chip: {}", caps.chip_generation));
    notes.push(format!("Unified memory: {} GB", gb));
    notes.push(format!("GPU layers: {}", gpu_layers));
    notes.push(format!("Context window: {} tokens", n_ctx));
    notes.push(format!("Threads: {}", n_threads));
    notes.push(format!(
        "Flash Attention: {}",
        if params.flash_attn {
            "enabled"
        } else {
            "disabled"
        }
    ));

    if caps.gpu_core_count > 0 {
        notes.push(format!("GPU cores: {}", caps.gpu_core_count));
    }
    notes.push(format!(
        "Memory bandwidth: {:.0} GB/s",
        caps.memory_bandwidth_gbs
    ));

    info!(
        chip = %caps.chip_generation,
        preset = ?preset,
        gpu_layers,
        n_ctx,
        "Metal optimizations applied"
    );

    MetalOptimizationReport {
        caps,
        preset,
        params,
        gpu_layers,
        notes,
    }
}

fn choose_preset(caps: &MetalCapabilities, gb: usize) -> MetalPreset {
    if !caps.metal_available {
        return MetalPreset::CpuOnly;
    }

    match &caps.chip_generation {
        AppleChipGeneration::M1Ultra
        | AppleChipGeneration::M2Ultra
        | AppleChipGeneration::M3Ultra
        | AppleChipGeneration::M1Max
        | AppleChipGeneration::M2Max
        | AppleChipGeneration::M3Max
        | AppleChipGeneration::M4Max => {
            if gb >= 32 {
                MetalPreset::MaxThroughput
            } else {
                MetalPreset::Balanced
            }
        }
        AppleChipGeneration::M1Pro
        | AppleChipGeneration::M2Pro
        | AppleChipGeneration::M3Pro
        | AppleChipGeneration::M4Pro
        | AppleChipGeneration::M2
        | AppleChipGeneration::M3
        | AppleChipGeneration::M4 => MetalPreset::Balanced,
        AppleChipGeneration::M1 => {
            if gb >= 16 {
                MetalPreset::Balanced
            } else {
                MetalPreset::MemorySaver
            }
        }
        _ => MetalPreset::MemorySaver,
    }
}

/// Print a human-readable summary of the Metal optimization report
pub fn print_report(report: &MetalOptimizationReport) {
    println!("\n╔══════════════════════════════════════════");
    println!("║  Barq Metal Backend Report");
    println!("╠══════════════════════════════════════════");
    for note in &report.notes {
        println!("║  ▸ {}", note);
    }
    println!("║  ▸ Preset: {:?}", report.preset);
    println!(
        "║  ▸ Inference score: {}/100",
        report.caps.inference_score()
    );
    println!("╚══════════════════════════════════════════\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tune_non_apple() {
        use crate::metal_detect::MetalCapabilities;
        use std::collections::HashMap;

        let caps = MetalCapabilities {
            is_apple: false,
            is_apple_silicon: false,
            metal_available: false,
            chip_generation: AppleChipGeneration::NonApple,
            unified_memory_bytes: 0,
            gpu_core_count: 0,
            ane_core_count: 0,
            memory_bandwidth_gbs: 0.0,
            supports_mps: false,
            max_recommended_ctx: 4096,
            metadata: HashMap::new(),
        };
        let report = tune_for_capabilities(caps, ContextParams::default());
        assert_eq!(report.preset, MetalPreset::CpuOnly);
        assert_eq!(report.gpu_layers, 0);
    }

    #[test]
    fn test_apply_metal_optimizations_runs() {
        // Should complete without panicking on any platform
        let report = apply_metal_optimizations(ContextParams::default());
        assert!(report.gpu_layers <= 9999);
    }
}
