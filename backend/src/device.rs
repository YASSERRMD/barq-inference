//! Device management and abstraction

use std::sync::Arc;

use core::error::{Error, Result};

/// Device type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// CUDA GPU device (NVIDIA)
    Cuda,
    /// Metal GPU device (Apple)
    Metal,
    /// ROCm GPU device (AMD)
    Rocm,
    /// Vulkan GPU device
    Vulkan,
    /// SYCL device (Intel)
    Sycl,
}

/// Device ID
pub type DeviceId = usize;

/// Device trait
pub trait Device: Send + Sync {
    /// Returns the device type
    fn device_type(&self) -> DeviceType;

    /// Returns the device ID
    fn device_id(&self) -> DeviceId;

    /// Returns the device name
    fn name(&self) -> &str;

    /// Returns total memory in bytes
    fn total_memory(&self) -> usize;

    /// Returns available memory in bytes
    fn available_memory(&self) -> usize;

    /// Synchronize device operations
    fn synchronize(&self) -> Result<()>;

    /// Clone the device
    fn clone_box(&self) -> Box<dyn Device>;
}

impl Clone for Box<dyn Device> {
    fn clone(&self) -> Box<dyn Device> {
        self.clone_box()
    }
}

/// CPU device implementation
#[derive(Debug, Clone)]
pub struct CpuDevice {
    id: DeviceId,
    name: String,
    num_threads: usize,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            id: 0,
            name: "cpu".to_string(),
            num_threads: rayon::current_num_threads(),
        }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        let mut device = Self::new();
        device.num_threads = num_threads;
        device
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for CpuDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn total_memory(&self) -> usize {
        // Get total system memory
        #[cfg(unix)]
        {
            unsafe {
                let mut info: libc::sysinfo = std::mem::zeroed();
                if libc::sysinfo(&mut info) == 0 {
                    return (info.totalram as usize) * (info.mem_unit as usize);
                }
            }
        }
        // Fallback
        16 * 1024 * 1024 * 1024 // 16 GB
    }

    fn available_memory(&self) -> usize {
        #[cfg(unix)]
        {
            unsafe {
                let mut info: libc::sysinfo = std::mem::zeroed();
                if libc::sysinfo(&mut info) == 0 {
                    return (info.freeram as usize) * (info.mem_unit as usize);
                }
            }
        }
        // Fallback
        8 * 1024 * 1024 * 1024 // 8 GB
    }

    fn synchronize(&self) -> Result<()> {
        // CPU is always synchronized
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Device> {
        Box::new(self.clone())
    }
}

/// GPU device placeholder
#[derive(Debug, Clone)]
pub struct GpuDevice {
    device_type: DeviceType,
    id: DeviceId,
    name: String,
    total_memory: usize,
}

impl GpuDevice {
    pub fn new(device_type: DeviceType, id: DeviceId, name: String, total_memory: usize) -> Self {
        Self {
            device_type,
            id,
            name,
            total_memory,
        }
    }
}

impl Device for GpuDevice {
    fn device_type(&self) -> DeviceType {
        self.device_type
    }

    fn device_id(&self) -> DeviceId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn total_memory(&self) -> usize {
        self.total_memory
    }

    fn available_memory(&self) -> usize {
        // TODO: Implement actual query
        self.total_memory / 2
    }

    fn synchronize(&self) -> Result<()> {
        // TODO: Implement actual synchronization
        Err(Error::Unsupported("GPU sync not implemented".to_string()))
    }

    fn clone_box(&self) -> Box<dyn Device> {
        Box::new(self.clone())
    }
}

/// Device manager
pub struct DeviceManager {
    devices: Vec<Box<dyn Device>>,
    default_device: Option<Box<dyn Device>>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            default_device: None,
        }
    }

    pub fn register(&mut self, device: Box<dyn Device>) {
        self.devices.push(device);
    }

    pub fn get_device(&self, id: DeviceId) -> Option<&dyn Device> {
        self.devices.get(id).map(|d| d.as_ref())
    }

    pub fn set_default(&mut self, device: Box<dyn Device>) {
        self.default_device = Some(device);
    }

    pub fn default_device(&self) -> Option<&dyn Device> {
        self.default_device.as_ref().map(|d| d.as_ref())
    }

    pub fn list_devices(&self) -> Vec<&dyn Device> {
        self.devices.iter().map(|d| d.as_ref()).collect()
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        let mut manager = Self::new();
        let cpu = Box::new(CpuDevice::new()) as Box<dyn Device>;
        manager.register(cpu);
        manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device() {
        let device = CpuDevice::new();
        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert_eq!(device.device_id(), 0);
    }

    #[test]
    fn test_device_manager() {
        let mut manager = DeviceManager::new();
        assert_eq!(manager.list_devices().len(), 0);

        let cpu = Box::new(CpuDevice::new()) as Box<dyn Device>;
        manager.register(cpu);
        assert_eq!(manager.list_devices().len(), 1);
    }
}
