use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::{accum4, clamp_coord};
use crate::gpu_shaders::spatial;

/// General NxN convolution with arbitrary kernel.
#[derive(Clone)]
pub struct Convolve {
    pub kernel: Vec<f32>,
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub divisor: f32,
}

impl Filter for Convolve {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let kw = self.kernel_width as usize;
        let kh = self.kernel_height as usize;
        let rw = kw / 2;
        let rh = kh / 2;
        let inv_div = 1.0 / self.divisor;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sx = clamp_coord(x as i32 + kx as i32 - rw as i32, w);
                        let sy = clamp_coord(y as i32 + ky as i32 - rh as i32, h);
                        let k = self.kernel[ky * kw + kx];
                        let idx = (sy * w + sx) * 4;
                        accum4(&mut sum, &input[idx..], k);
                    }
                }
                let out_idx = (y * w + x) * 4;
                out[out_idx] = sum[0] * inv_div;
                out[out_idx + 1] = sum[1] * inv_div;
                out[out_idx + 2] = sum[2] * inv_div;
                out[out_idx + 3] = sum[3] * inv_div;
            }
        }

        Ok(out)
    }
}

// ── Convolve GPU (single-pass with kernel extra_buffer) ─────────────────────

impl GpuFilter for Convolve {
    fn shader_body(&self) -> &str { spatial::CONVOLVE }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&self.kernel_width.to_le_bytes());
        buf.extend_from_slice(&self.kernel_height.to_le_bytes());
        buf.extend_from_slice(&(1.0f32 / self.divisor).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Pack kernel weights as f32 bytes
        let mut kb = Vec::with_capacity(self.kernel.len() * 4);
        for &w in &self.kernel {
            kb.extend_from_slice(&w.to_le_bytes());
        }
        vec![kb]
    }
}
