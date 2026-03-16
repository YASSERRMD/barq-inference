//! Softmax implementation

use core::tensor::{Tensor, TensorType, Shape, TensorData};
use core::error::{Error, Result};

pub fn softmax(logits: &[f32]) -> Result<Vec<f32>> {
    if logits.is_empty() {
        return Ok(Vec::new());
    }

    // Find max for numerical stability
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

    // Compute exp and sum
    let mut exp_sum = 0.0f32;
    let mut exp_values = Vec::with_capacity(logits.len());

    for &logit in logits {
        let exp = (logit - max_logit).exp();
        exp_sum += exp;
        exp_values.push(exp);
    }

    // Normalize
    let result: Vec<f32> = exp_values.iter().map(|&e| e / exp_sum).collect();

    Ok(result)
}

pub fn softmax_inplace(logits: &mut [f32]) -> Result<()> {
    let output = softmax(logits)?;
    logits.copy_from_slice(&output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = softmax(&input).unwrap();

        // Check probabilities sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check all values are in [0, 1]
        for &p in &result {
            assert!(p >= 0.0 && p <= 1.0);
        }

        // Check ordering is preserved
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }
    }

    #[test]
    fn test_softmax_empty() {
        let input: Vec<f32> = vec![];
        let result = softmax(&input).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_softmax_inplace() {
        let mut input = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut input).unwrap();

        let sum: f32 = input.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
