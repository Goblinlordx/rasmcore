use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::gpu_shaders::spatial;

/// Displacement map — warp pixels by per-pixel offset fields.
///
/// `map_x` and `map_y` are f32 slices of absolute source coordinates.
#[derive(Clone)]
pub struct DisplacementMap {
    pub map_x: Vec<f32>,
    pub map_y: Vec<f32>,
}

impl Filter for DisplacementMap {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        if self.map_x.len() != n || self.map_y.len() != n {
            return Err(PipelineError::InvalidParams(format!(
                "displacement map size mismatch: expected {n}, got x={} y={}",
                self.map_x.len(),
                self.map_y.len()
            )));
        }

        let mut out = vec![0.0f32; n * 4];
        let wi = w as i32;
        let hi = h as i32;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let sx = self.map_x[idx];
                let sy = self.map_y[idx];

                // Bilinear interpolation
                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let sample = |px: i32, py: i32, c: usize| -> f32 {
                    if px >= 0 && px < wi && py >= 0 && py < hi {
                        input[(py as usize * w + px as usize) * 4 + c]
                    } else {
                        0.0
                    }
                };

                let out_off = idx * 4;
                for c in 0..4 {
                    let v = sample(x0, y0, c) * (1.0 - fx) * (1.0 - fy)
                        + sample(x0 + 1, y0, c) * fx * (1.0 - fy)
                        + sample(x0, y0 + 1, c) * (1.0 - fx) * fy
                        + sample(x0 + 1, y0 + 1, c) * fx * fy;
                    out[out_off + c] = v;
                }
            }
        }

        Ok(out)
    }
}

// ── DisplacementMap GPU (single-pass with map extra_buffer) ─────────────────

impl GpuFilter for DisplacementMap {
    fn shader_body(&self) -> &str { spatial::DISPLACEMENT_MAP }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Interleave map_x and map_y as f32 pairs
        let mut buf = Vec::with_capacity(self.map_x.len() * 8);
        for (&mx, &my) in self.map_x.iter().zip(self.map_y.iter()) {
            buf.extend_from_slice(&mx.to_le_bytes());
            buf.extend_from_slice(&my.to_le_bytes());
        }
        vec![buf]
    }
}
