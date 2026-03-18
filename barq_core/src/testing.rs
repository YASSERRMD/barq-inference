//! Testing utilities for barq-inference
//!
//! Provides common test helpers, fixtures, and utilities:
//! - Model fixtures for testing
//! - Tensor comparison helpers
//! - Performance measurement utilities
//! - Test data generators

use crate::error::Result;
use crate::tensor::{Shape, Tensor, TensorData, TensorType};

/// Test fixture for common tensor operations
pub struct TensorFixture;

impl TensorFixture {
    /// Create a simple 1D tensor
    pub fn simple_1d() -> Tensor {
        Tensor::new(
            None,
            TensorType::F32,
            Shape::vector(4),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap()
    }

    /// Create a simple 2D tensor (matrix)
    pub fn simple_2d() -> Tensor {
        Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(2, 2),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap()
    }

    /// Create identity matrix
    pub fn identity(n: usize) -> Tensor {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }

        Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(n, n),
            TensorData::F32(data),
        )
        .unwrap()
    }

    /// Create random tensor with values in [0, 1]
    pub fn random(shape: Shape) -> Tensor {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut data = Vec::with_capacity(shape.num_elements());
        let mut rng = seed;

        for _ in 0..shape.num_elements() {
            // Simple LCG for deterministic pseudo-randomness
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = (rng >> 32) as f32 / u32::MAX as f32;
            data.push(value);
        }

        Tensor::new(None, TensorType::F32, shape, TensorData::F32(data)).unwrap()
    }

    /// Create tensor filled with specific value
    pub fn filled(shape: Shape, value: f32) -> Tensor {
        let data = vec![value; shape.num_elements()];
        Tensor::new(None, TensorType::F32, shape, TensorData::F32(data)).unwrap()
    }

    /// Create sequential tensor [0, 1, 2, ...]
    pub fn sequential(shape: Shape) -> Tensor {
        let data: Vec<f32> = (0..shape.num_elements()).map(|i| i as f32).collect();

        Tensor::new(None, TensorType::F32, shape, TensorData::F32(data)).unwrap()
    }
}

/// Tensor comparison helpers
pub struct TensorAssertions;

impl TensorAssertions {
    /// Assert two tensors are approximately equal
    pub fn assert_close(a: &Tensor, b: &Tensor, tolerance: f32) {
        assert_eq!(
            a.shape().dims(),
            b.shape().dims(),
            "Tensor shapes differ: {:?} vs {:?}",
            a.shape().dims(),
            b.shape().dims()
        );

        let a_data = a.as_f32_slice().unwrap();
        let b_data = b.as_f32_slice().unwrap();

        for (i, (a_val, b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
            let diff = (a_val - b_val).abs();
            assert!(
                diff <= tolerance,
                "Tensor values differ at index {}: {} vs {} (diff: {} > {})",
                i,
                a_val,
                b_val,
                diff,
                tolerance
            );
        }
    }

    /// Assert tensor shape
    pub fn assert_shape(tensor: &Tensor, expected: &[usize]) {
        assert_eq!(
            tensor.shape().dims(),
            expected,
            "Expected shape {:?}, got {:?}",
            expected,
            tensor.shape().dims()
        );
    }

    /// Assert tensor has specific value at index
    pub fn assert_value_at(tensor: &Tensor, index: usize, expected: f32, tolerance: f32) {
        let data = tensor.as_f32_slice().unwrap();
        let actual = data[index];
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "Expected {} at index {}, got {} (diff: {})",
            expected,
            index,
            actual,
            diff
        );
    }
}

/// Performance measurement utilities
pub struct BenchmarkTimer {
    start: std::time::Instant,
}

impl BenchmarkTimer {
    /// Start a new benchmark timer
    pub fn start() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Get elapsed time in seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Measure execution time of a function
    pub fn measure<F, R>(f: F) -> (R, f64)
    where
        F: FnOnce() -> R,
    {
        let timer = Self::start();
        let result = f();
        let elapsed = timer.elapsed_secs();
        (result, elapsed)
    }
}

/// Statistics helpers for test results
pub struct TestStats {
    values: Vec<f64>,
}

impl TestStats {
    /// Create new stats collection
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Add a value
    pub fn add(&mut self, value: f64) {
        self.values.push(value);
    }

    /// Get mean
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }

        let mean = self.mean();
        let variance = self.values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (self.values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Get min
    pub fn min(&self) -> f64 {
        self.values.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// Get max
    pub fn max(&self) -> f64 {
        self.values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Get median
    pub fn median(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        }
    }

    /// Get percentile (0-100)
    pub fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Get number of samples
    pub fn count(&self) -> usize {
        self.values.len()
    }
}

impl Default for TestStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_fixture() {
        let tensor = TensorFixture::simple_2d();
        TensorAssertions::assert_shape(&tensor, &[2, 2]);

        let identity = TensorFixture::identity(3);
        TensorAssertions::assert_shape(&identity, &[3, 3]);

        let filled = TensorFixture::filled(Shape::vector(5), 7.0);
        TensorAssertions::assert_shape(&filled, &[5]);
    }

    #[test]
    fn test_tensor_assertions() {
        let a = TensorFixture::simple_2d();
        let b = TensorFixture::simple_2d();

        TensorAssertions::assert_close(&a, &b, 1e-6);
        TensorAssertions::assert_value_at(&a, 0, 1.0, 1e-6);
        TensorAssertions::assert_value_at(&a, 3, 4.0, 1e-6);
    }

    #[test]
    fn test_benchmark_timer() {
        let (result, time) = BenchmarkTimer::measure(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(time >= 0.01); // At least 10ms
    }

    #[test]
    fn test_test_stats() {
        let mut stats = TestStats::new();
        stats.add(1.0);
        stats.add(2.0);
        stats.add(3.0);
        stats.add(4.0);
        stats.add(5.0);

        assert_eq!(stats.count(), 5);
        assert_eq!(stats.mean(), 3.0);
        assert_eq!(stats.median(), 3.0);
        assert_eq!(stats.min(), 1.0);
        assert_eq!(stats.max(), 5.0);
        assert_eq!(stats.percentile(50.0), 3.0);
        assert_eq!(stats.percentile(90.0), 4.0); // 90th percentile of [1,2,3,4,5] is 4

        // Check std_dev (approx 1.58 for this data)
        let std_dev = stats.std_dev();
        assert!((std_dev - 1.58).abs() < 0.01);
    }
}
