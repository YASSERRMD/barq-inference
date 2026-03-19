//! Fused Mixture-of-Experts helpers.
//!
//! This module keeps the implementation CPU-only and tensor-based so it can be
//! reused by higher level MoE wrappers without introducing backend cycles.

use barq_core::error::{Error, Result};
use barq_core::ops::{BinaryOp, MatMul, Silu, UnaryOp};
use barq_core::tensor::{Shape, Tensor, TensorData, TensorType};
use std::cmp::Ordering;

/// Configuration for fused MoE execution.
#[derive(Debug, Clone)]
pub struct MoEFusedConfig {
    /// Total number of experts in the model.
    pub n_experts: usize,
    /// Number of experts selected per token before reduction.
    pub n_expert_per_token: usize,
    /// Minimum routing weight kept by smart expert reduction.
    pub reduction_threshold: f32,
}

impl Default for MoEFusedConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            n_expert_per_token: 2,
            reduction_threshold: 0.05,
        }
    }
}

/// Grouped token batch for a single expert.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertBatch {
    pub expert_id: usize,
    pub token_indices: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Fused MoE execution helpers.
pub struct MoEFusedOps;

impl MoEFusedOps {
    /// Fused FFN up/gate path: `silu(input @ gate) * (input @ up)`.
    pub fn fused_up_gate(
        input: &Tensor,
        gate_weight: &Tensor,
        up_weight: &Tensor,
    ) -> Result<Tensor> {
        let gate = MatMul.apply(input, gate_weight)?;
        let up = MatMul.apply(input, up_weight)?;
        let activated = Silu.apply(&gate)?;

        if activated.shape() != up.shape() {
            return Err(Error::dimension_mismatch(
                "MoE fused gate/up projections must have matching shapes",
            ));
        }

        match (activated.dtype(), up.dtype()) {
            (TensorType::F32, TensorType::F32) => {
                let activated_data = activated.as_f32_slice()?;
                let up_data = up.as_f32_slice()?;
                let result: Vec<f32> = activated_data
                    .iter()
                    .zip(up_data.iter())
                    .map(|(a, b)| a * b)
                    .collect();

                Tensor::new(
                    None,
                    TensorType::F32,
                    activated.shape().clone(),
                    TensorData::F32(result),
                )
            }
            _ => Err(Error::Unsupported(
                "Fused MoE gate/up only supports F32 tensors".to_string(),
            )),
        }
    }

    /// Full fused FFN: up/gate, activation, down projection.
    pub fn fused_ffn(
        input: &Tensor,
        gate_weight: &Tensor,
        up_weight: &Tensor,
        down_weight: &Tensor,
    ) -> Result<Tensor> {
        let gated = Self::fused_up_gate(input, gate_weight, up_weight)?;
        MatMul.apply(&gated, down_weight)
    }

    /// Convert routing logits into top-k expert assignments with softmax weights.
    pub fn select_top_k_experts(
        routing_logits: &Tensor,
        n_expert_per_token: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        if routing_logits.ndim() != 2 {
            return Err(Error::tensor("Routing logits must be a 2D tensor"));
        }

        if routing_logits.dtype() != TensorType::F32 {
            return Err(Error::Unsupported(
                "Routing logits must be F32 for fused MoE routing".to_string(),
            ));
        }

        let rows = routing_logits.shape().dims()[0];
        let cols = routing_logits.shape().dims()[1];
        if cols == 0 {
            return Err(Error::tensor(
                "Routing logits must contain at least one expert",
            ));
        }

        let k = n_expert_per_token.clamp(1, cols);
        let data = routing_logits.as_f32_slice()?;

        let mut expert_ids = Vec::with_capacity(rows);
        let mut expert_weights = Vec::with_capacity(rows);

        for row_idx in 0..rows {
            let row = &data[row_idx * cols..(row_idx + 1) * cols];
            let mut ranked: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            ranked.truncate(k);

            let max_logit = ranked
                .iter()
                .map(|(_, value)| *value)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = ranked
                .iter()
                .map(|(_, value)| (*value - max_logit).exp())
                .collect();
            let denom = exp_scores.iter().sum::<f32>().max(f32::EPSILON);

            expert_ids.push(ranked.iter().map(|(idx, _)| *idx).collect());
            expert_weights.push(exp_scores.into_iter().map(|v| v / denom).collect());
        }

        Ok((expert_ids, expert_weights))
    }

    /// Apply smart expert reduction to prune low-value experts.
    pub fn reduce_expert_assignments(
        expert_ids: &[Vec<usize>],
        expert_weights: &[Vec<f32>],
        max_active_experts: usize,
        min_weight: f32,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let mut reduced_ids = Vec::with_capacity(expert_ids.len());
        let mut reduced_weights = Vec::with_capacity(expert_weights.len());

        for (ids, weights) in expert_ids.iter().zip(expert_weights.iter()) {
            let mut ranked: Vec<(usize, f32)> = ids
                .iter()
                .copied()
                .zip(weights.iter().copied())
                .filter(|(_, weight)| *weight >= min_weight)
                .collect();

            if ranked.is_empty() {
                if let Some((idx, weight)) = ids
                    .iter()
                    .copied()
                    .zip(weights.iter().copied())
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                {
                    ranked.push((idx, weight));
                }
            }

            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            ranked.truncate(max_active_experts.max(1));

            let weight_sum = ranked
                .iter()
                .map(|(_, weight)| *weight)
                .sum::<f32>()
                .max(f32::EPSILON);
            reduced_ids.push(ranked.iter().map(|(idx, _)| *idx).collect());
            reduced_weights.push(
                ranked
                    .iter()
                    .map(|(_, weight)| weight / weight_sum)
                    .collect(),
            );
        }

        (reduced_ids, reduced_weights)
    }

    /// Group token assignments by expert for batched dispatch.
    pub fn batch_by_expert(
        expert_ids: &[Vec<usize>],
        expert_weights: &[Vec<f32>],
        n_experts: usize,
    ) -> Vec<ExpertBatch> {
        let mut batches: Vec<ExpertBatch> = (0..n_experts)
            .map(|expert_id| ExpertBatch {
                expert_id,
                token_indices: Vec::new(),
                weights: Vec::new(),
            })
            .collect();

        for (token_idx, (ids, weights)) in expert_ids.iter().zip(expert_weights.iter()).enumerate()
        {
            for (local_idx, &expert_id) in ids.iter().enumerate() {
                if let Some(batch) = batches.get_mut(expert_id) {
                    batch.token_indices.push(token_idx);
                    batch.weights.push(weights[local_idx]);
                }
            }
        }

        batches
            .into_iter()
            .filter(|batch| !batch.token_indices.is_empty())
            .collect()
    }

    /// Dispatch tokens to experts in batches and accumulate weighted outputs.
    pub fn dispatch_experts(
        input: &Tensor,
        expert_ffns: &[Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>],
        expert_ids: &[Vec<usize>],
        expert_weights: &[Vec<f32>],
    ) -> Result<Tensor> {
        if input.ndim() != 2 {
            return Err(Error::tensor("MoE dispatch expects a 2D input tensor"));
        }

        if input.dtype() != TensorType::F32 {
            return Err(Error::Unsupported(
                "MoE dispatch currently supports only F32 inputs".to_string(),
            ));
        }

        if expert_ids.len() != expert_weights.len() {
            return Err(Error::tensor(
                "Expert ids and weights must contain the same number of tokens",
            ));
        }

        if input.shape().dims()[0] != expert_ids.len() {
            return Err(Error::dimension_mismatch(
                "Routing assignments must match the input batch size",
            ));
        }

        if expert_ffns.is_empty() {
            return Err(Error::tensor("At least one expert function is required"));
        }

        let hidden = input.shape().dims()[1];
        let input_data = input.as_f32_slice()?;
        let mut output = vec![0.0f32; input.shape().num_elements()];
        let batches = Self::batch_by_expert(expert_ids, expert_weights, expert_ffns.len());

        for batch in batches {
            let mut batched_input = Vec::with_capacity(batch.token_indices.len() * hidden);
            for &token_idx in &batch.token_indices {
                let start = token_idx * hidden;
                batched_input.extend_from_slice(&input_data[start..start + hidden]);
            }

            let batch_tensor = Tensor::new(
                None,
                TensorType::F32,
                Shape::matrix(batch.token_indices.len(), hidden),
                TensorData::F32(batched_input),
            )?;

            let batch_output = expert_ffns
                .get(batch.expert_id)
                .ok_or_else(|| Error::tensor("Expert id out of range"))?(
                &batch_tensor
            )?;

            if batch_output.dtype() != TensorType::F32
                || batch_output.shape() != batch_tensor.shape()
            {
                return Err(Error::tensor(
                    "Expert outputs must be F32 tensors with the same batch shape",
                ));
            }

            let batch_data = batch_output.as_f32_slice()?;
            for (row_idx, &token_idx) in batch.token_indices.iter().enumerate() {
                let weight = batch.weights[row_idx];
                let out_row = &mut output[token_idx * hidden..(token_idx + 1) * hidden];
                let expert_row = &batch_data[row_idx * hidden..(row_idx + 1) * hidden];
                for (dst, src) in out_row.iter_mut().zip(expert_row.iter()) {
                    *dst += src * weight;
                }
            }
        }

        Tensor::new(
            None,
            TensorType::F32,
            input.shape().clone(),
            TensorData::F32(output),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use barq_core::testing::{BenchmarkTimer, TensorAssertions};

    fn tensor_from_rows(rows: usize, cols: usize, values: Vec<f32>) -> Tensor {
        Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(rows, cols),
            TensorData::F32(values),
        )
        .unwrap()
    }

    #[test]
    fn test_select_top_k_experts() {
        let logits = tensor_from_rows(2, 4, vec![0.1, 2.0, 1.0, -1.0, 3.0, 0.0, 1.5, 1.0]);
        let (ids, weights) = MoEFusedOps::select_top_k_experts(&logits, 2).unwrap();

        assert_eq!(ids, vec![vec![1, 2], vec![0, 2]]);
        assert_eq!(weights.len(), 2);
        assert!((weights[0][0] + weights[0][1] - 1.0).abs() < 1e-6);
        assert!((weights[1][0] + weights[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_expert_assignments() {
        let ids = vec![vec![0, 1, 2], vec![2, 1, 0]];
        let weights = vec![vec![0.7, 0.2, 0.1], vec![0.4, 0.35, 0.25]];

        let (reduced_ids, reduced_weights) =
            MoEFusedOps::reduce_expert_assignments(&ids, &weights, 2, 0.15);

        assert_eq!(reduced_ids, vec![vec![0, 1], vec![2, 1]]);
        assert_eq!(reduced_weights.len(), 2);
        assert!((reduced_weights[0][0] + reduced_weights[0][1] - 1.0).abs() < 1e-6);
        assert!((reduced_weights[1][0] + reduced_weights[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dispatch_experts() {
        let input = tensor_from_rows(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
                    TensorData::F32(values),
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
                    TensorData::F32(values),
                )
            }),
        ];

        let expert_ids = vec![vec![0], vec![1]];
        let expert_weights = vec![vec![1.0], vec![1.0]];

        let output =
            MoEFusedOps::dispatch_experts(&input, &expert_ffns, &expert_ids, &expert_weights)
                .unwrap();

        assert_eq!(output.shape().dims(), &[2, 3]);
        assert_eq!(
            output.as_f32_slice().unwrap(),
            &[2.0, 4.0, 6.0, 12.0, 15.0, 18.0]
        );
    }

    fn naive_dispatch(
        input: &Tensor,
        expert_ffns: &[Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>],
        expert_ids: &[Vec<usize>],
        expert_weights: &[Vec<f32>],
    ) -> Result<Tensor> {
        let hidden = input.shape().dims()[1];
        let input_data = input.as_f32_slice()?;
        let mut output = vec![0.0f32; input.shape().num_elements()];

        for (token_idx, (ids, weights)) in expert_ids.iter().zip(expert_weights.iter()).enumerate()
        {
            let token = tensor_from_rows(
                1,
                hidden,
                input_data[token_idx * hidden..(token_idx + 1) * hidden].to_vec(),
            );

            for (expert_id, weight) in ids.iter().zip(weights.iter()) {
                let expert_output = expert_ffns[*expert_id](&token)?;
                let row = expert_output.as_f32_slice()?;
                let out_row = &mut output[token_idx * hidden..(token_idx + 1) * hidden];
                for (dst, src) in out_row.iter_mut().zip(row.iter()) {
                    *dst += src * *weight;
                }
            }
        }

        Tensor::new(
            None,
            TensorType::F32,
            input.shape().clone(),
            TensorData::F32(output),
        )
    }

    #[test]
    fn benchmark_dispatch_experts_speedup() {
        let input = tensor_from_rows(32, 16, (0..512).map(|i| (i as f32 % 13.0) / 13.0).collect());

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
                    TensorData::F32(values),
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
                    TensorData::F32(values),
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
                    TensorData::F32(values),
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
                    TensorData::F32(values),
                )
            }),
        ];

        let expert_ids: Vec<Vec<usize>> = (0..32)
            .map(|token_idx| vec![token_idx % 4, (token_idx + 1) % 4])
            .collect();
        let expert_weights: Vec<Vec<f32>> = (0..32).map(|_| vec![0.7, 0.3]).collect();

        let (fused_output, fused_secs) = BenchmarkTimer::measure(|| {
            MoEFusedOps::dispatch_experts(&input, &expert_ffns, &expert_ids, &expert_weights)
                .unwrap()
        });
        let (naive_output, naive_secs) = BenchmarkTimer::measure(|| {
            naive_dispatch(&input, &expert_ffns, &expert_ids, &expert_weights).unwrap()
        });

        TensorAssertions::assert_close(&fused_output, &naive_output, 1e-6);
        println!(
            "MoE fused dispatch benchmark: fused={:.3}ms naive={:.3}ms speedup={:.2}x",
            fused_secs * 1000.0,
            naive_secs * 1000.0,
            if fused_secs > 0.0 {
                naive_secs / fused_secs
            } else {
                0.0
            }
        );
    }
}
