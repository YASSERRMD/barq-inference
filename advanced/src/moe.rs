//! Mixture of Experts (MoE) inference
//!
//! Efficient implementation for MoE models like Mixtral,
//! supporting expert routing and load balancing.

use barq_core::error::{Error, Result};
use barq_core::tensor::Tensor;
use models::MoEFusedOps;

/// MoE configuration
#[derive(Debug, Clone)]
pub struct MoEConfig {
    /// Number of experts
    pub n_experts: usize,
    /// Number of experts per token
    pub n_expert_per_token: usize,
    /// Load balancing loss coefficient
    pub aux_loss_coef: f32,
    /// Smart expert reduction threshold
    pub ser_threshold: f32,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            n_expert_per_token: 2,
            aux_loss_coef: 0.01,
            ser_threshold: 0.05,
        }
    }
}

/// MoE router that selects experts for each token
pub struct MoERouter {
    config: MoEConfig,
}

impl MoERouter {
    pub fn new(config: MoEConfig) -> Self {
        Self { config }
    }

    /// Route tokens to experts
    ///
    /// Returns: (expert_ids, expert_weights) for each token
    pub fn route(&self, routing_logits: &Tensor) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        let (expert_ids, expert_weights) =
            MoEFusedOps::select_top_k_experts(routing_logits, self.config.n_expert_per_token)?;

        // Clamp expert ids to the configured expert pool so malformed metadata does not
        // accidentally address an out-of-range expert.
        let mut clamped_ids = Vec::with_capacity(expert_ids.len());
        let mut clamped_weights = Vec::with_capacity(expert_weights.len());

        for (ids, weights) in expert_ids.into_iter().zip(expert_weights.into_iter()) {
            let mut row_ids = Vec::with_capacity(ids.len());
            let mut row_weights = Vec::with_capacity(weights.len());

            for (expert_id, weight) in ids.into_iter().zip(weights.into_iter()) {
                row_ids.push(expert_id % self.config.n_experts.max(1));
                row_weights.push(weight);
            }

            clamped_ids.push(row_ids);
            clamped_weights.push(row_weights);
        }

        Ok((clamped_ids, clamped_weights))
    }

    /// Compute load balancing loss
    pub fn compute_aux_loss(&self, expert_counts: &[usize]) -> f32 {
        let mean_count = expert_counts.iter().sum::<usize>() as f32 / expert_counts.len() as f32;

        let variance: f32 = expert_counts
            .iter()
            .map(|&c| (c as f32 - mean_count).powi(2))
            .sum();

        self.config.aux_loss_coef * variance / expert_counts.len() as f32
    }
}

/// MoE inference engine
pub struct MoEInference {
    config: MoEConfig,
    router: MoERouter,
}

impl MoEInference {
    pub fn new(config: MoEConfig) -> Self {
        let router = MoERouter::new(config.clone());
        Self { config, router }
    }

    fn derive_routing_logits(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 2 {
            return Err(Error::tensor(
                "MoE inference expects a 2D [tokens, hidden] tensor",
            ));
        }

        let rows = input.shape().dims()[0];
        let hidden = input.shape().dims()[1];
        let mut logits = Vec::with_capacity(rows * self.config.n_experts);
        let data = input.as_f32_slice()?;

        for row_idx in 0..rows {
            let row = &data[row_idx * hidden..(row_idx + 1) * hidden];
            let row_mean = if row.is_empty() {
                0.0
            } else {
                row.iter().sum::<f32>() / row.len() as f32
            };

            for expert_idx in 0..self.config.n_experts {
                let base = row.get(expert_idx % hidden).copied().unwrap_or(row_mean);
                logits.push(base + row_mean * 0.01 + expert_idx as f32 * 1e-3);
            }
        }

        Tensor::new(
            None,
            barq_core::tensor::TensorType::F32,
            barq_core::tensor::Shape::matrix(rows, self.config.n_experts),
            barq_core::tensor::TensorData::F32(logits),
        )
    }

    /// Process tokens through MoE layers
    pub fn forward(
        &self,
        input: &Tensor,
        expert_ffns: &[Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>],
    ) -> Result<Tensor> {
        let routing_logits = self.derive_routing_logits(input)?;
        self.forward_with_routing(input, &routing_logits, expert_ffns)
    }

    /// Process tokens through MoE layers with explicit routing logits.
    pub fn forward_with_routing(
        &self,
        input: &Tensor,
        routing_logits: &Tensor,
        expert_ffns: &[Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>],
    ) -> Result<Tensor> {
        if expert_ffns.len() != self.config.n_experts {
            return Err(Error::tensor(format!(
                "Expected {} experts, got {}",
                self.config.n_experts,
                expert_ffns.len()
            )));
        }

        let (expert_ids, expert_weights) = self.router.route(routing_logits)?;
        let (expert_ids, expert_weights) = MoEFusedOps::reduce_expert_assignments(
            &expert_ids,
            &expert_weights,
            self.config.n_expert_per_token,
            self.config.ser_threshold,
        );

        MoEFusedOps::dispatch_experts(input, expert_ffns, &expert_ids, &expert_weights)
    }

    /// Returns the router
    pub fn router(&self) -> &MoERouter {
        &self.router
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use barq_core::TensorType;

    #[test]
    fn test_moe_config() {
        let config = MoEConfig::default();
        assert_eq!(config.n_experts, 8);
        assert_eq!(config.n_expert_per_token, 2);
        assert!((config.ser_threshold - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_moe_router() {
        let config = MoEConfig::default();
        let router = MoERouter::new(config);

        // Create dummy routing logits
        let logits = Tensor::new(
            None,
            TensorType::F32,
            barq_core::tensor::Shape::matrix(2, 4),
            barq_core::tensor::TensorData::F32(vec![0.1, 2.0, 1.0, -1.0, 3.0, 0.0, 1.5, 1.0]),
        )
        .unwrap();

        let result = router.route(&logits);
        assert!(result.is_ok());

        let (expert_ids, weights) = result.unwrap();
        assert_eq!(expert_ids.len(), 2);
        assert_eq!(weights.len(), 2);
        assert_eq!(expert_ids[0], vec![1, 2]);
        assert_eq!(expert_ids[1], vec![0, 2]);
    }

    #[test]
    fn test_moe_forward_with_routing() {
        let config = MoEConfig::default();
        let moe = MoEInference::new(config);

        let input = Tensor::new(
            None,
            TensorType::F32,
            barq_core::tensor::Shape::matrix(2, 3),
            barq_core::tensor::TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();

        let routing_logits = Tensor::new(
            None,
            TensorType::F32,
            barq_core::tensor::Shape::matrix(2, 8),
            barq_core::tensor::TensorData::F32(vec![
                0.1, 2.0, 1.0, -1.0, 0.2, 0.0, -0.5, -1.5, 3.0, 0.0, 1.5, 1.0, -1.0, 0.5, 0.1, -0.5,
            ]),
        )
        .unwrap();

        let expert_ffns: Vec<Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>> = vec![
            Box::new(|tensor: &Tensor| {
                let values = tensor
                    .as_f32_slice()?
                    .iter()
                    .map(|v| v * 2.0)
                    .collect::<Vec<_>>();
                Tensor::new(
                    None,
                    TensorType::F32,
                    tensor.shape().clone(),
                    barq_core::tensor::TensorData::F32(values),
                )
            }),
            Box::new(|tensor: &Tensor| {
                let values = tensor
                    .as_f32_slice()?
                    .iter()
                    .map(|v| v * 3.0)
                    .collect::<Vec<_>>();
                Tensor::new(
                    None,
                    TensorType::F32,
                    tensor.shape().clone(),
                    barq_core::tensor::TensorData::F32(values),
                )
            }),
            Box::new(|tensor: &Tensor| {
                let values = tensor
                    .as_f32_slice()?
                    .iter()
                    .map(|v| v * 4.0)
                    .collect::<Vec<_>>();
                Tensor::new(
                    None,
                    TensorType::F32,
                    tensor.shape().clone(),
                    barq_core::tensor::TensorData::F32(values),
                )
            }),
            Box::new(|tensor: &Tensor| {
                let values = tensor
                    .as_f32_slice()?
                    .iter()
                    .map(|v| v * 5.0)
                    .collect::<Vec<_>>();
                Tensor::new(
                    None,
                    TensorType::F32,
                    tensor.shape().clone(),
                    barq_core::tensor::TensorData::F32(values),
                )
            }),
            Box::new(|tensor: &Tensor| {
                let values = tensor
                    .as_f32_slice()?
                    .iter()
                    .map(|v| v * 6.0)
                    .collect::<Vec<_>>();
                Tensor::new(
                    None,
                    TensorType::F32,
                    tensor.shape().clone(),
                    barq_core::tensor::TensorData::F32(values),
                )
            }),
            Box::new(|tensor: &Tensor| {
                let values = tensor
                    .as_f32_slice()?
                    .iter()
                    .map(|v| v * 7.0)
                    .collect::<Vec<_>>();
                Tensor::new(
                    None,
                    TensorType::F32,
                    tensor.shape().clone(),
                    barq_core::tensor::TensorData::F32(values),
                )
            }),
            Box::new(|tensor: &Tensor| {
                let values = tensor
                    .as_f32_slice()?
                    .iter()
                    .map(|v| v * 8.0)
                    .collect::<Vec<_>>();
                Tensor::new(
                    None,
                    TensorType::F32,
                    tensor.shape().clone(),
                    barq_core::tensor::TensorData::F32(values),
                )
            }),
            Box::new(|tensor: &Tensor| {
                let values = tensor
                    .as_f32_slice()?
                    .iter()
                    .map(|v| v * 9.0)
                    .collect::<Vec<_>>();
                Tensor::new(
                    None,
                    TensorType::F32,
                    tensor.shape().clone(),
                    barq_core::tensor::TensorData::F32(values),
                )
            }),
        ];

        let output = moe
            .forward_with_routing(&input, &routing_logits, &expert_ffns)
            .unwrap();

        assert_eq!(output.shape().dims(), &[2, 3]);
        assert!(output
            .as_f32_slice()
            .unwrap()
            .iter()
            .all(|value| *value > 0.0));
    }
}
