//! Phase 7.1 — Apple Silicon / Metal Runtime Detection
//!
//! Detects at runtime whether the host machine has Apple Silicon (M-series)
//! and whether Metal GPU acceleration is available. Provides rich capability
//! information to guide the inference engine's backend selection.
//!
//! Does **not** require the `metal` feature flag — detection is done via
//! platform queries so it is always safe to call on any target.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Chip generation for Apple Silicon
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppleChipGeneration {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M3Ultra,
    M4,
    M4Pro,
    M4Max,
    /// Intel Mac or non-Apple Silicon
    Intel,
    /// Unknown Apple ARM chip
    UnknownAppleArm,
    /// Non-Apple platform
    NonApple,
}

impl AppleChipGeneration {
    /// Estimated peak GPU TFLOPs (FP32) for this chip
    pub fn gpu_tflops(&self) -> f32 {
        match self {
            Self::M1 => 2.6,
            Self::M1Pro => 5.2,
            Self::M1Max => 10.4,
            Self::M1Ultra => 20.8,
            Self::M2 => 3.6,
            Self::M2Pro => 6.8,
            Self::M2Max => 13.6,
            Self::M2Ultra => 27.2,
            Self::M3 => 3.6,
            Self::M3Pro => 7.4,
            Self::M3Max => 14.2,
            Self::M3Ultra => 28.4,
            Self::M4 => 4.6,
            Self::M4Pro => 10.0,
            Self::M4Max => 20.0,
            _ => 0.0,
        }
    }

    /// Recommended GPU layers to offload for typical 7B model
    pub fn recommended_gpu_layers_7b(&self) -> u32 {
        match self {
            Self::Intel | Self::NonApple => 0,
            Self::M1 => 22,
            Self::M1Pro | Self::M2 => 32, // up to ~16GB
            Self::M1Max | Self::M2Pro | Self::M3 | Self::M4 => 32,
            Self::M1Ultra | Self::M2Max | Self::M3Pro | Self::M4Pro => 32,
            Self::M2Ultra | Self::M3Max | Self::M4Max | Self::M3Ultra => 32,
            _ => 20,
        }
    }
}

impl std::fmt::Display for AppleChipGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::M1 => "Apple M1",
            Self::M1Pro => "Apple M1 Pro",
            Self::M1Max => "Apple M1 Max",
            Self::M1Ultra => "Apple M1 Ultra",
            Self::M2 => "Apple M2",
            Self::M2Pro => "Apple M2 Pro",
            Self::M2Max => "Apple M2 Max",
            Self::M2Ultra => "Apple M2 Ultra",
            Self::M3 => "Apple M3",
            Self::M3Pro => "Apple M3 Pro",
            Self::M3Max => "Apple M3 Max",
            Self::M3Ultra => "Apple M3 Ultra",
            Self::M4 => "Apple M4",
            Self::M4Pro => "Apple M4 Pro",
            Self::M4Max => "Apple M4 Max",
            Self::Intel => "Intel x86_64 Mac",
            Self::UnknownAppleArm => "Unknown Apple ARM",
            Self::NonApple => "Non-Apple Platform",
        };
        write!(f, "{}", s)
    }
}

/// Complete Metal capability report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalCapabilities {
    /// Whether this is an Apple platform
    pub is_apple: bool,
    /// Whether Apple Silicon (ARM) is present
    pub is_apple_silicon: bool,
    /// Whether Metal GPU support is available
    pub metal_available: bool,
    /// Detected chip generation
    pub chip_generation: AppleChipGeneration,
    /// Unified memory size in bytes (0 if unknown)
    pub unified_memory_bytes: usize,
    /// Number of GPU cores (0 if unknown)
    pub gpu_core_count: u32,
    /// Number of Neural Engine cores (ANE)
    pub ane_core_count: u32,
    /// Peak GPU memory bandwidth GB/s
    pub memory_bandwidth_gbs: f32,
    /// Whether Metal Performance Shaders (MPS) are supported
    pub supports_mps: bool,
    /// Maximum recommended context size for this hardware
    pub max_recommended_ctx: usize,
    /// Extra platform metadata
    pub metadata: HashMap<String, String>,
}

impl MetalCapabilities {
    /// Returns a score 0-100 indicating inference suitability
    pub fn inference_score(&self) -> u8 {
        if !self.metal_available {
            return 10; // Pure CPU
        }
        let tflops = self.chip_generation.gpu_tflops();
        let score = (tflops / 28.4 * 90.0) as u8 + 10;
        score.min(100)
    }

    /// Whether all-GPU offload is recommended
    pub fn should_offload_all_layers(&self) -> bool {
        matches!(
            self.chip_generation,
            AppleChipGeneration::M1Max
                | AppleChipGeneration::M1Ultra
                | AppleChipGeneration::M2Max
                | AppleChipGeneration::M2Ultra
                | AppleChipGeneration::M3Max
                | AppleChipGeneration::M3Ultra
                | AppleChipGeneration::M4Max
        ) && self.unified_memory_bytes >= 32 * 1024 * 1024 * 1024
    }
}

/// Run Metal / Apple Silicon detection
pub fn detect() -> MetalCapabilities {
    detect_inner()
}

#[cfg(target_os = "macos")]
fn detect_inner() -> MetalCapabilities {
    use std::process::Command;

    let is_apple = true;

    // Detect architecture from sysctl
    let arch = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let is_apple_silicon = !arch.to_lowercase().contains("intel");

    // Chip model
    let chip_str = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_lowercase();

    let chip_generation = parse_chip_generation(&chip_str, is_apple_silicon);

    // Unified memory
    let unified_memory_bytes = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(0);

    // GPU core count (via system_profiler or sysctl)
    let gpu_core_count = Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            // Parse "GPU Core Count: 30" style lines
            s.lines()
                .find(|l| l.contains("GPU Core Count"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|n| n.trim().parse::<u32>().ok())
        })
        .unwrap_or(0);

    // ANE cores (heuristic — M-series all have 16 ANE cores)
    let ane_core_count = if is_apple_silicon { 16 } else { 0 };

    // Memory bandwidth (heuristic from known specs)
    let memory_bandwidth_gbs = match &chip_generation {
        AppleChipGeneration::M1 => 68.25,
        AppleChipGeneration::M1Pro => 200.0,
        AppleChipGeneration::M1Max => 400.0,
        AppleChipGeneration::M1Ultra => 800.0,
        AppleChipGeneration::M2 => 100.0,
        AppleChipGeneration::M2Pro => 200.0,
        AppleChipGeneration::M2Max => 400.0,
        AppleChipGeneration::M2Ultra => 800.0,
        AppleChipGeneration::M3 => 100.0,
        AppleChipGeneration::M3Pro => 150.0,
        AppleChipGeneration::M3Max => 300.0,
        AppleChipGeneration::M3Ultra => 600.0,
        AppleChipGeneration::M4 => 120.0,
        AppleChipGeneration::M4Pro => 273.0,
        AppleChipGeneration::M4Max => 546.0,
        _ => 50.0,
    };

    let metal_available = is_apple; // Metal available on all modern Macs
    let max_recommended_ctx = (unified_memory_bytes / (1024 * 1024 * 1024)).max(1) * 8192;
    let supports_mps = is_apple_silicon;

    let mut metadata = HashMap::new();
    metadata.insert("arch_string".into(), arch.trim().to_string());
    metadata.insert(
        "unified_memory_gb".into(),
        format!("{:.1}", unified_memory_bytes as f64 / (1 << 30) as f64),
    );

    let caps = MetalCapabilities {
        is_apple,
        is_apple_silicon,
        metal_available,
        chip_generation,
        unified_memory_bytes,
        gpu_core_count,
        ane_core_count,
        memory_bandwidth_gbs,
        supports_mps,
        max_recommended_ctx,
        metadata,
    };

    info!(
        chip = %caps.chip_generation,
        metal = caps.metal_available,
        memory_gb = caps.unified_memory_bytes / (1 << 30),
        gpu_cores = caps.gpu_core_count,
        "Metal detection complete"
    );

    caps
}

#[cfg(not(target_os = "macos"))]
fn detect_inner() -> MetalCapabilities {
    MetalCapabilities {
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
    }
}

fn parse_chip_generation(brand: &str, is_apple_silicon: bool) -> AppleChipGeneration {
    if !is_apple_silicon {
        return AppleChipGeneration::Intel;
    }

    // Try parsing via system_profiler chip name
    let chip_name_raw = std::process::Command::new("system_profiler")
        .args(["SPHardwareDataType"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .to_lowercase();

    let chip = if chip_name_raw.contains("m1 ultra") {
        AppleChipGeneration::M1Ultra
    } else if chip_name_raw.contains("m1 max") {
        AppleChipGeneration::M1Max
    } else if chip_name_raw.contains("m1 pro") {
        AppleChipGeneration::M1Pro
    } else if chip_name_raw.contains("m1") {
        AppleChipGeneration::M1
    } else if chip_name_raw.contains("m2 ultra") {
        AppleChipGeneration::M2Ultra
    } else if chip_name_raw.contains("m2 max") {
        AppleChipGeneration::M2Max
    } else if chip_name_raw.contains("m2 pro") {
        AppleChipGeneration::M2Pro
    } else if chip_name_raw.contains("m2") {
        AppleChipGeneration::M2
    } else if chip_name_raw.contains("m3 ultra") {
        AppleChipGeneration::M3Ultra
    } else if chip_name_raw.contains("m3 max") {
        AppleChipGeneration::M3Max
    } else if chip_name_raw.contains("m3 pro") {
        AppleChipGeneration::M3Pro
    } else if chip_name_raw.contains("m3") {
        AppleChipGeneration::M3
    } else if chip_name_raw.contains("m4 max") {
        AppleChipGeneration::M4Max
    } else if chip_name_raw.contains("m4 pro") {
        AppleChipGeneration::M4Pro
    } else if chip_name_raw.contains("m4") {
        AppleChipGeneration::M4
    } else if is_apple_silicon {
        AppleChipGeneration::UnknownAppleArm
    } else {
        AppleChipGeneration::Intel
    };

    chip
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_runs_without_panic() {
        let caps = detect();
        // Must always return a valid struct regardless of platform
        println!("{:?}", caps);
    }

    #[test]
    fn test_chip_display() {
        assert_eq!(AppleChipGeneration::M2Max.to_string(), "Apple M2 Max");
        assert_eq!(
            AppleChipGeneration::NonApple.to_string(),
            "Non-Apple Platform"
        );
    }

    #[test]
    fn test_inference_score_cpu_only() {
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
        assert_eq!(caps.inference_score(), 10);
    }

    #[test]
    fn test_gpu_tflops() {
        assert!(AppleChipGeneration::M3Max.gpu_tflops() > AppleChipGeneration::M3.gpu_tflops());
    }
}
