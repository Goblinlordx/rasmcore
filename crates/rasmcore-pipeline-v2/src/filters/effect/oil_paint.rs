use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{gpu_params_push_u32, luminance};

/// Oil paint — neighborhood mode filter (most frequent intensity bin).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "oil_paint", category = "effect", cost = "O(n * radius^2)")]
pub struct OilPaint {
    #[param(min = 1, max = 20, default = 4)]
    pub radius: u32,
}

impl Filter for OilPaint {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as usize;
        const BINS: usize = 256;
        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let mut count = [0u32; BINS];
                let mut sum_r = [0.0f32; BINS];
                let mut sum_g = [0.0f32; BINS];
                let mut sum_b = [0.0f32; BINS];

                let y0 = y.saturating_sub(r);
                let y1 = (y + r + 1).min(h);
                let x0 = x.saturating_sub(r);
                let x1 = (x + r + 1).min(w);

                for ny in y0..y1 {
                    for nx in x0..x1 {
                        let idx = (ny * w + nx) * 4;
                        let intensity = luminance(input[idx], input[idx + 1], input[idx + 2]);
                        let bin = ((intensity.max(0.0) * 255.0) as usize).min(BINS - 1);
                        count[bin] += 1;
                        sum_r[bin] += input[idx];
                        sum_g[bin] += input[idx + 1];
                        sum_b[bin] += input[idx + 2];
                    }
                }

                // Find most frequent bin
                let max_bin = count
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, c)| *c)
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let cnt = count[max_bin].max(1) as f32;
                let idx = (y * w + x) * 4;
                out[idx] = sum_r[max_bin] / cnt;
                out[idx + 1] = sum_g[max_bin] / cnt;
                out[idx + 2] = sum_b[max_bin] / cnt;
            }
        }
        Ok(out)
    }
}

impl GpuFilter for OilPaint {
    fn shader_body(&self) -> &str {
        include_str!("../../shaders/oil_paint.wgsl")
    }
    fn workgroup_size(&self) -> [u32; 3] { [8, 8, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, self.radius);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}
