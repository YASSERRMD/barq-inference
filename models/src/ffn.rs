//! Feed-forward network implementation

use barq_core::tensor::{Tensor, TensorType, Shape, TensorData};
use barq_core::error::{Error, Result};
use barq_core::ops::{MatMul, Add, BinaryOp};
use barq_core::normalization;

/// Feed-forward network (MLP)
pub struct FeedForward {
    hidden_dim: usize,
    output_dim: usize,
}

impl FeedForward {
    pub fn new(hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            hidden_dim,
            output_dim,
        }
    }

    /// Forward pass through FFN
    pub fn forward(
        &self,
        input: &Tensor,
        gate_weight: &Tensor,
        up_weight: &Tensor,
        down_weight: &Tensor,
    ) -> Result<Tensor> {
        // gate: input @ gate_weight
        let gate = MatMul.apply(input, gate_weight)?;

        // up: input @ up_weight
        let up = MatMul.apply(input, up_weight)?;

        // Apply activation to gate (SiLU)
        let activated = self.silu(&gate)?;

        // Element-wise multiply: silu(gate) * up
        let gated = self.mul(&activated, &up)?;

        // Down projection: gated @ down_weight
        let output = MatMul.apply(&gated, down_weight)?;

        Ok(output)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        match x.dtype() {
            TensorType::F32 => {
                let data = x.as_f32_slice()?;
                let result: Vec<f32> = data.iter().map(|&v| {
                    v / (1.0 + (-v).exp())
                }).collect();

                Tensor::new(
                    None,
                    TensorType::F32,
                    x.shape().clone(),
                    TensorData::F32(result),
                )
            }
            _ => Err(Error::Unsupported(format!("SiLU not implemented for {:?}", x.dtype()))),
        }
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(Error::tensor("Shape mismatch for element-wise multiplication"));
        }

        match (a.dtype(), b.dtype()) {
            (TensorType::F32, TensorType::F32) => {
                let a_data = a.as_f32_slice()?;
                let b_data = b.as_f32_slice()?;
                let result: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).collect();

                Tensor::new(
                    None,
                    TensorType::F32,
                    a.shape().clone(),
                    TensorData::F32(result),
                )
            }
            _ => Err(Error::Unsupported("Mul not implemented for non-f32 types".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward_creation() {
        let ffn = FeedForward::new(11008, 4096);
        assert_eq!(ffn.hidden_dim, 11008);
        assert_eq!(ffn.output_dim, 4096);
    }
}
