//! Q5_K quantization implementation
//!
//! 5-bit quantization using the llama.cpp super-block layout.
//! Block size: QK_K (256)
//! Effective bits per weight: 5.5

use barq_core::error::{Error, Result};
use half::f16;

pub const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

/// Q5_K block structure matching llama.cpp `block_q5_K`.
///
/// Layout:
/// - `d`: f16 super-block scale for quantized scales
/// - `dmin`: f16 super-block scale for quantized mins
/// - `scales[12]`: packed 6-bit scale/min values
/// - `qh[32]`: high bit planes for the 5-bit quants
/// - `qs[128]`: low 4 bits for the 5-bit quants
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

impl BlockQ5K {
    pub const SIZE_BYTES: usize = 2 + 2 + K_SCALE_SIZE + (QK_K / 8) + (QK_K / 2);

    pub const fn size_bytes() -> usize {
        Self::SIZE_BYTES
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE_BYTES] {
        let mut out = [0u8; Self::SIZE_BYTES];
        out[0..2].copy_from_slice(&self.d.to_le_bytes());
        out[2..4].copy_from_slice(&self.dmin.to_le_bytes());
        out[4..16].copy_from_slice(&self.scales);
        out[16..48].copy_from_slice(&self.qh);
        out[48..176].copy_from_slice(&self.qs);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::SIZE_BYTES {
            return Err(Error::Quantization(format!(
                "Q5_K block must be {} bytes, got {}",
                Self::SIZE_BYTES,
                bytes.len()
            )));
        }

        let mut scales = [0u8; K_SCALE_SIZE];
        scales.copy_from_slice(&bytes[4..16]);

        let mut qh = [0u8; QK_K / 8];
        qh.copy_from_slice(&bytes[16..48]);

        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&bytes[48..176]);

        Ok(Self {
            d: u16::from_le_bytes([bytes[0], bytes[1]]),
            dmin: u16::from_le_bytes([bytes[2], bytes[3]]),
            scales,
            qh,
            qs,
        })
    }

    pub fn dequantize(&self, output: &mut [f32]) {
        let d = f16::from_bits(self.d).to_f32();
        let min = f16::from_bits(self.dmin).to_f32();

        let mut out_idx = 0usize;
        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;

        for q_offset in (0..QK_K).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, &self.scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;

            let (sc, m) = get_scale_min_k4(is + 1, &self.scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;

            let ql = &self.qs[q_offset / 2..q_offset / 2 + 32];

            for l in 0..32 {
                let q = (ql[l] & 0x0f) as u8 + if self.qh[l] & u1 != 0 { 16 } else { 0 };
                if out_idx < output.len() {
                    output[out_idx] = d1 * q as f32 - m1;
                    out_idx += 1;
                }
            }

            for l in 0..32 {
                let q = (ql[l] >> 4) as u8 + if self.qh[l] & u2 != 0 { 16 } else { 0 };
                if out_idx < output.len() {
                    output[out_idx] = d2 * q as f32 - m2;
                    out_idx += 1;
                }
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    pub fn quantize(data: &[f32; QK_K]) -> Self {
        let mut scales = [0u8; K_SCALE_SIZE];
        let mut qh = [0u8; QK_K / 8];
        let mut qs = [0u8; QK_K / 2];

        let mut block_scales = [0.0f32; QK_K / 32];
        let mut block_mins = [0.0f32; QK_K / 32];
        let mut block_weights = [0.0f32; QK_K / 32];
        let mut sub_weights = [0.0f32; 32];
        let mut qvals = [0u8; QK_K];
        let mut sub_qvals = [0u8; 32];

        let sum_x2: f32 = data.iter().map(|v| v * v).sum();
        let sigma2 = 2.0 * sum_x2 / QK_K as f32;
        let av_x = sigma2.sqrt();

        for j in 0..(QK_K / 32) {
            let start = j * 32;
            let block = &data[start..start + 32];

            for l in 0..32 {
                sub_weights[l] = av_x + block[l].abs();
            }

            block_weights[j] = sub_weights.iter().sum();
            block_scales[j] = make_qkx3_quants(
                32,
                31,
                block,
                &sub_weights,
                &mut qvals[start..start + 32],
                &mut block_mins[j],
                &mut sub_qvals,
                -0.9,
                0.05,
                36,
                false,
            );
        }

        let d_block = make_qp_quants(QK_K / 32, 63, &block_scales, &block_weights);
        let m_block = make_qp_quants(QK_K / 32, 63, &block_mins, &block_weights);

        let d_block_bits = f16::from_f32(d_block).to_bits();
        let m_block_bits = f16::from_f32(m_block).to_bits();

        for j in 0..(QK_K / 32) {
            let ls = if d_block == 0.0 {
                0
            } else {
                ((block_scales[j] / d_block).round().clamp(0.0, 63.0)) as u8
            };
            let lm = if m_block == 0.0 {
                0
            } else {
                ((block_mins[j] / m_block).round().clamp(0.0, 63.0)) as u8
            };

            if j < 4 {
                scales[j] = ls;
                scales[j + 4] = lm;
            } else {
                scales[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
                scales[j - 4] |= (ls >> 4) << 6;
                scales[j] |= (lm >> 4) << 6;
            }
        }

        for j in 0..(QK_K / 32) {
            let (sc, m) = get_scale_min_k4(j, &scales);
            let d = d_block * sc as f32;
            let dm = m_block * m as f32;
            if d == 0.0 {
                continue;
            }

            let start = j * 32;
            for ii in 0..32 {
                let l = ((data[start + ii] + dm) / d).round().clamp(0.0, 31.0) as u8;
                qvals[start + ii] = l;
            }
        }

        qh.fill(0);

        let mut m1 = 1u8;
        let mut m2 = 2u8;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let mut l1 = qvals[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }

                let mut l2 = qvals[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }

                qs[n / 2 + j] = l1 | (l2 << 4);
            }

            m1 <<= 2;
            m2 <<= 2;
        }

        Self {
            d: d_block_bits,
            dmin: m_block_bits,
            scales,
            qh,
            qs,
        }
    }
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

#[inline]
fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}

fn make_qkx3_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];

    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }

    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        l[..n].fill(0);
        *the_min = -min;
        return 0.0;
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0;

    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let diff = scale * l[i] as f32 + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        best_mad += weights[i] * diff;
    }

    if nstep < 1 {
        *the_min = -min;
        return scale;
    }

    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0;
        let mut sum_l2 = 0.0;
        let mut sum_xl = 0.0;

        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let w = weights[i];
            let qf = q as f32;
            sum_l += w * qf;
            sum_l2 += w * qf * qf;
            sum_xl += w * qf * x[i];
        }

        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }

            let mut mad = 0.0;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                mad += weights[i] * diff;
            }

            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }

    *the_min = -min;
    scale
}

fn make_qp_quants(n: usize, nmax: i32, x: &[f32], weights: &[f32]) -> f32 {
    let mut max: f32 = 0.0;
    for i in 0..n {
        max = max.max(x[i]);
    }

    if max < 1e-15 {
        return 0.0;
    }

    let mut iscale = nmax as f32 / max;
    let mut l = vec![0u8; n];
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]).clamp(0, nmax) as u8;
    }

    let mut scale = 1.0 / iscale;
    let mut best_mse = 0.0;
    for i in 0..n {
        let diff = x[i] - scale * l[i] as f32;
        best_mse += weights[i] * diff * diff;
    }

    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0;
        for i in 0..n {
            let q = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * q as f32;
            mse += weights[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    let mut sumlx = 0.0;
    let mut suml2 = 0.0;
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).min(nmax) as u8;
        l[i] = q;
        let w = weights[i];
        let qf = q as f32;
        sumlx += w * x[i] * qf;
        suml2 += w * qf * qf;
    }

    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = weights[i];
            let slx = sumlx - w * x[i] * l[i] as f32;
            let sl2 = suml2 - w * (l[i] as f32) * (l[i] as f32);
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l as u8 != l[i] {
                    let slx_new = slx + w * x[i] * new_l as f32;
                    let sl2_new = sl2 + w * (new_l as f32) * (new_l as f32);
                    if slx_new * slx_new * suml2 > sumlx * sumlx * sl2_new {
                        l[i] = new_l as u8;
                        sumlx = slx_new;
                        suml2 = sl2_new;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }

    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

pub struct Q5K {
    block_size: usize,
}

impl Q5K {
    pub fn new() -> Self {
        Self { block_size: QK_K }
    }

    pub fn quantize(&self, input: &[f32]) -> Result<Vec<u8>> {
        let n_blocks = input.len() / QK_K;
        let remainder = input.len() % QK_K;
        let total_blocks = if remainder > 0 {
            n_blocks + 1
        } else {
            n_blocks
        };

        let mut output = Vec::with_capacity(total_blocks * BlockQ5K::size_bytes());

        for block_idx in 0..n_blocks {
            let start = block_idx * QK_K;
            let block: &[f32; QK_K] = input[start..start + QK_K]
                .try_into()
                .map_err(|_| Error::Quantization("Invalid block size".into()))?;

            let qblock = BlockQ5K::quantize(block);
            output.extend_from_slice(&qblock.to_bytes());
        }

        if remainder > 0 {
            let mut last_block = [0.0f32; QK_K];
            last_block[..remainder].copy_from_slice(&input[n_blocks * QK_K..]);
            let qblock = BlockQ5K::quantize(&last_block);
            output.extend_from_slice(&qblock.to_bytes());
        }

        Ok(output)
    }

    pub fn dequantize(&self, input: &[u8], output_size: usize) -> Result<Vec<f32>> {
        let block_bytes = BlockQ5K::size_bytes();
        let n_blocks = (output_size + QK_K - 1) / QK_K;

        let mut output = vec![0.0f32; n_blocks * QK_K];

        for (block_idx, out_chunk) in output.chunks_mut(QK_K).enumerate().take(n_blocks) {
            let offset = block_idx * block_bytes;
            if offset + block_bytes > input.len() {
                break;
            }

            let qblock = BlockQ5K::from_bytes(&input[offset..offset + block_bytes])?;
            qblock.dequantize(out_chunk);
        }

        output.truncate(output_size);
        Ok(output)
    }
}

impl Default for Q5K {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q5_k_block_size() {
        let q = Q5K::new();
        assert_eq!(q.block_size, QK_K);
    }

    #[test]
    fn test_q5_k_size() {
        assert_eq!(BlockQ5K::size_bytes(), 176);
    }

    #[test]
    fn test_q5_k_roundtrip() {
        let quant = Q5K::new();

        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();
        let quantized = quant.quantize(&input).unwrap();
        let dequantized = quant.dequantize(&quantized, input.len()).unwrap();

        assert_eq!(dequantized.len(), input.len());

        for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error <= 0.1,
                "Error at index {}: {} vs {} (error={})",
                i,
                orig,
                deq,
                error
            );
        }
    }

    #[test]
    fn test_q5_k_constant_block_layout() {
        let mut raw = [0u8; BlockQ5K::SIZE_BYTES];
        raw[0..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        raw[2..4].copy_from_slice(&f16::from_f32(0.0).to_bits().to_le_bytes());
        raw[4..16].fill(1);
        raw[48..].fill(0x11);

        let block = BlockQ5K::from_bytes(&raw).unwrap();
        let mut output = [0.0f32; QK_K];
        block.dequantize(&mut output);

        assert!(output.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }
}
