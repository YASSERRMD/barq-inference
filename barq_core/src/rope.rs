//! Rotary Position Embedding (RoPE) implementation

use crate::error::{Error, Result};

/// Compute rotary position embeddings
pub fn rope(
    positions: &[usize],
    dim: usize,
    freq_base: f32,
    scale: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if !dim.is_multiple_of(2) {
        return Err(Error::tensor("RoPE dimension must be even"));
    }

    let half_dim = dim / 2;
    let mut cos = Vec::with_capacity(positions.len() * dim);
    let mut sin = Vec::with_capacity(positions.len() * dim);

    // Precompute frequencies
    let mut freqs = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let theta = freq_base * (2.0 * i as f32 / dim as f32).exp2();
        freqs.push(theta);
    }

    // Compute cos and sin for each position
    for &pos in positions {
        let scaled_pos = pos as f32 * scale;

        for i in 0..half_dim {
            let angle = scaled_pos / freqs[i];
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }

    Ok((cos, sin))
}

/// Apply rotary embeddings to query and key
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    dim: usize,
) -> Result<()> {
    let seq_len = q.len() / dim;

    if cos.len() != seq_len * (dim / 2) || sin.len() != seq_len * (dim / 2) {
        return Err(Error::tensor(format!(
            "Cos/sin size mismatch: expected {}, got {}",
            seq_len * (dim / 2),
            cos.len()
        )));
    }

    for i in 0..seq_len {
        for j in 0..(dim / 2) {
            let q_real_idx = i * dim + j;
            let q_imag_idx = i * dim + j + dim / 2;
            let k_real_idx = i * dim + j;
            let k_imag_idx = i * dim + j + dim / 2;

            let c = cos[i * (dim / 2) + j];
            let s = sin[i * (dim / 2) + j];

            // Rotate query
            let q_real = q[q_real_idx];
            let q_imag = q[q_imag_idx];
            q[q_real_idx] = q_real * c - q_imag * s;
            q[q_imag_idx] = q_real * s + q_imag * c;

            // Rotate key
            let k_real = k[k_real_idx];
            let k_imag = k[k_imag_idx];
            k[k_real_idx] = k_real * c - k_imag * s;
            k[k_imag_idx] = k_real * s + k_imag * c;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope() {
        let positions = vec![0, 1, 2];
        let dim = 64;

        let (cos, sin) = rope(&positions, dim, 10000.0, 1.0).unwrap();

        assert_eq!(cos.len(), positions.len() * (dim / 2));
        assert_eq!(sin.len(), positions.len() * (dim / 2));

        // Check that cos^2 + sin^2 = 1
        for i in 0..cos.len() {
            let val = cos[i] * cos[i] + sin[i] * sin[i];
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_rope() {
        let mut q = vec![1.0f32; 128];
        let mut k = vec![2.0f32; 128];

        let positions = vec![0, 1];
        let dim = 64;

        let (cos, sin) = rope(&positions, dim, 10000.0, 1.0).unwrap();

        apply_rope(&mut q, &mut k, &cos, &sin, dim).unwrap();

        // Values should have changed
        assert!(q.iter().any(|&x| x != 1.0));
        assert!(k.iter().any(|&x| x != 2.0));
    }

    #[test]
    fn test_rope_odd_dim() {
        let positions = vec![0, 1];
        let dim = 63;

        let result = rope(&positions, dim, 10000.0, 1.0);
        assert!(result.is_err());
    }
}
