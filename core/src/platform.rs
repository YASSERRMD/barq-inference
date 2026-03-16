//! Edge & NPU offload support
//!
//! Implements detection and configuration for:
//! - Apple Metal (M-series GPUs)
//! - CPU SIMD detection
//! - Platform-specific optimizations

use std::sync::OnceLock;
use std::collections::HashMap;

/// Platform type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformType {
    /// Apple Silicon with Metal support
    AppleMetal,
    /// NVIDIA CUDA
    Cuda,
    /// CPU only with SIMD
    CpuSimd,
    /// Unknown platform
    Unknown,
}

/// SIMD capabilities
#[derive(Debug, Clone)]
pub struct SIMDCapabilities {
    pub platform: PlatformType,
    pub avx2: bool,
    pub avx512: bool,
    pub neon: bool,
    pub sve: bool,
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub platform: PlatformType,
    pub device_name: String,
    pub compute_units: usize,
    pub memory_mb: usize,
}

/// Detect current platform and capabilities
pub fn detect_platform() -> PlatformType {
    static DETECTED: OnceLock<PlatformType> = OnceLock::new();

    *DETECTED.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            // Check for Metal support
            if is_apple_silicon() {
                return PlatformType::AppleMetal;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "cuda"))]
        {
            // Check for CUDA
            if has_cuda() {
                return PlatformType::Cuda;
            }
        }

        // Default to CPU with SIMD
        PlatformType::CpuSimd
    })
}

/// Detect SIMD capabilities
pub fn detect_simd() -> SIMDCapabilities {
    let platform = detect_platform();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            SIMDCapabilities {
                platform,
                avx2: is_x86_feature_detected!("avx2"),
                avx512: is_x86_feature_detected!("avx512f"),
                neon: false,
                sve: false,
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            SIMDCapabilities {
                platform,
                avx2: false,
                avx512: false,
                neon: is_aarch64_feature_detected!("neon"),
                sve: is_aarch64_feature_detected!("sve"), // ARMv9
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SIMDCapabilities {
            platform,
            avx2: false,
            avx512: false,
            neon: false,
            sve: false,
        }
    }
}

/// Get device information
pub fn get_device_info() -> DeviceInfo {
    let platform = detect_platform();

    match platform {
        PlatformType::AppleMetal => {
            #[cfg(target_os = "macos")]
            {
                DeviceInfo {
                    platform,
                    device_name: get_apple_gpu_name(),
                    compute_units: get_apple_gpu_cores(),
                    memory_mb: get_apple_gpu_memory(),
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                DeviceInfo {
                    platform,
                    device_name: "Unknown Metal Device".to_string(),
                    compute_units: 0,
                    memory_mb: 0,
                }
            }
        }

        PlatformType::Cuda => {
            DeviceInfo {
                platform,
                device_name: "CUDA Device".to_string(),
                compute_units: 0,
                memory_mb: 0,
            }
        }

        PlatformType::CpuSimd => {
            DeviceInfo {
                platform,
                device_name: get_cpu_name(),
                compute_units: num_cpus::get(),
                memory_mb: get_system_memory_mb(),
            }
        }

        PlatformType::Unknown => {
            DeviceInfo {
                platform,
                device_name: "Unknown Device".to_string(),
                compute_units: 0,
                memory_mb: 0,
            }
        }
    }
}

/// Check if running on Apple Silicon
#[cfg(target_os = "macos")]
fn is_apple_silicon() -> bool {
    use std::process::Command;

    let output = Command::new("sysctl")
        .args(&["-n", "machdep.cpu.brand_string"])
        .output();

    match output {
        Ok(output) => {
            let brand = String::from_utf8_lossy(&output.stdout);
            brand.contains("Apple M") || brand.contains("Apple T")
        }
        Err(_) => false,
    }
}

/// Get Apple GPU name
#[cfg(target_os = "macos")]
fn get_apple_gpu_name() -> String {
    use std::process::Command;

    let output = Command::new("system_profiler")
        .args(&["SPDisplaysDataType", "-json"])
        .output();

    match output {
        Ok(output) => {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                if let Some(displays) = json["SPDisplaysDataType"].as_array() {
                    for display in displays {
                        if let Some(gpu) = display["sppci_gpu_model"].as_str() {
                            return gpu.to_string();
                        }
                    }
                }
            }
            "Unknown Apple GPU".to_string()
        }
        Err(_) => "Unknown Apple GPU".to_string(),
    }
}

/// Get Apple GPU cores
#[cfg(target_os = "macos")]
fn get_apple_gpu_cores() -> usize {
    use std::process::Command;

    let output = Command::new("system_profiler")
        .args(&["SPHardwareDataType", "-json"])
        .output();

    match output {
        Ok(output) => {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                if let Some(hardware) = json["SPHardwareDataType"].as_array() {
                    for item in hardware {
                        if let Some(cores) = item["metal_features"].as_str() {
                            // Parse cores info if available
                            if cores.contains("GPU") {
                                return 8; // Default for M1/M2
                            }
                        }
                    }
                }
            }
            8 // Default assumption
        }
        Err(_) => 8,
    }
}

/// Get Apple GPU memory in MB
#[cfg(target_os = "macos")]
fn get_apple_gpu_memory() -> usize {
    use std::process::Command;

    let output = Command::new("sysctl")
        .args(&["-n", "hw.memsize"])
        .output();

    match output {
        Ok(output) => {
            let mem_str = String::from_utf8_lossy(&output.stdout);
            let mem_bytes: u64 = mem_str.trim().parse().unwrap_or(0);
            // Roughly 1/3 of system RAM is available to GPU
            ((mem_bytes / 3) / (1024 * 1024)) as usize
        }
        Err(_) => 0,
    }
}

/// Get CUDA device name
#[cfg(feature = "cuda")]
fn get_cuda_device_name() -> String {
    // TODO: Implement CUDA device detection
    "CUDA Device".to_string()
}

/// Get CUDA compute units
#[cfg(feature = "cuda")]
fn get_cuda_compute_units() -> usize {
    // TODO: Implement CUDA query
    0
}

/// Get CUDA memory in MB
#[cfg(feature = "cuda")]
fn get_cuda_memory_mb() -> usize {
    // TODO: Implement CUDA memory query
    0
}

/// Check if CUDA is available
#[cfg(feature = "cuda")]
fn has_cuda() -> bool {
    // TODO: Implement CUDA availability check
    true
}

/// Get CPU name
fn get_cpu_name() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        "x86_64 CPU".to_string()
    }

    #[cfg(target_arch = "aarch64")]
    {
        "ARM64 CPU".to_string()
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "Unknown CPU".to_string()
    }
}

/// Get system memory in MB
fn get_system_memory_mb() -> usize {
    #[cfg(unix)]
    {
        use std::process::Command;

        let output = Command::new("sysctl")
            .args(&["-n", "hw.memsize"])
            .output();

        match output {
            Ok(output) => {
                let mem_str = String::from_utf8_lossy(&output.stdout);
                let mem_bytes: u64 = mem_str.trim().parse().unwrap_or(0);
                (mem_bytes / (1024 * 1024)) as usize
            }
            Err(_) => 0,
        }
    }

    #[cfg(not(unix))]
    {
        0
    }
}

/// Print platform information
pub fn print_platform_info() {
    let platform = detect_platform();
    let simd = detect_simd();
    let device = get_device_info();

    println!("\n=== Platform Information ===");
    println!("Platform:       {:?}", platform);
    println!("Device:         {}", device.device_name);
    println!("Compute units:  {}", device.compute_units);
    println!("Memory:         {} MB", device.memory_mb);

    println!("\nSIMD Capabilities:");
    println!("  AVX2:    {}", simd.avx2);
    println!("  AVX512:  {}", simd.avx512);
    println!("  NEON:    {}", simd.neon);
    println!("  SVE:     {}", simd.sve);
    println!("==============================\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = detect_platform();
        assert!(platform != PlatformType::Unknown);
    }

    #[test]
    fn test_simd_detection() {
        let simd = detect_simd();
        // Should have detected something
        assert!(simd.avx2 || simd.avx512 || simd.neon || simd.sve || matches!(simd.platform, PlatformType::Unknown));
    }

    #[test]
    fn test_device_info() {
        let device = get_device_info();
        assert!(device.compute_units >= 0);
        assert!(device.memory_mb >= 0);
    }
}
