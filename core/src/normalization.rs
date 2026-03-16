//! Layer normalization implementation

use core::tensor::{Tensor, TensorType, Shape, TensorData};
use core::error::{Error, Result};

pub fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let n = input.len();

    if weight.len() != n {
        return Err(Error::tensor(format!(
            "Weight size mismatch: expected {}, got {}",
            n, weight.len()
        )));
    }

    if bias.len() != n {
        return Err(Error::tensor(format!(
            "Bias size mismatch: expected {}, got {}",
            n, bias.len()
        )));
    }

    // Compute mean
    let sum: f32 = input.iter().sum();
    let mean = sum / n as f32;

    // Compute variance
    let variance = input.iter().map(|&x| {
        let diff = x - mean;
        diff * diff
    }).sum::<f32>() / n as f32;

    // Normalize and apply affine transform
    let result: Vec<f32> = input.iter().enumerate().map(|(i, &x)| {
        let normalized = (x - mean) / (variance + eps).sqrt();
        normalized * weight[i] + bias[i]
    }).collect();

    Ok(result)
}

pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let n = input.len();

    if weight.len() != n {
        return Err(Error::tensor(format!(
            "Weight size mismatch: expected {}, got {}",
            n, weight.len()
        )));
    }

    // Compute RMS (root mean square)
    let mean_square = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
    let rms = (mean_square + eps).sqrt();

    // Normalize and apply weight
    let result: Vec<f32> = input.iter().enumerate().map(|(i, &x)| {
        (x / rms) * weight[i]
    }).collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let result = layer_norm(&input, &weight, &bias, 1e-5).unwrap();

        assert_eq!(result.len(), input.len());

        // Check that result has approximately zero mean and unit variance
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        let variance = result.iter().map(|&x| {
            let diff = x - mean;
            diff * diff
        }).sum::<f32>() / result.len() as f32;

        assert!(mean.abs() < 1e-4);
        assert!((variance - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        let result = rms_norm(&input, &weight, 1e-5).unwrap();

        assert_eq!(result.len(), input.len());

        // Verify RMS of result is approximately 1.0 (due to weight=1.0)
        let mean_square = result.iter().map(|&x| x * x).sum::<f32>() / result.len() as f32;
        let rms = mean_square.sqrt();
        assert!((rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_with_weight_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let bias = vec![1.0, 1.0, 1.0, 1.0];

        let result = layer_norm(&input, &weight, &bias, 1e-5).unwrap();

        // With weight=2 and bias=1, result should be approximately 2*normalized + 1
        assert_eq!(result.len(), input.len());
    }
}
