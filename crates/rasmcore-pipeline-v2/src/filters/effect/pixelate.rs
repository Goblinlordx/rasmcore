use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::gpu_params_push_u32;

/// Pixelate — block-grid mosaic with mean color per cell.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "pixelate", category = "effect", cost = "O(n)")]
pub struct Pixelate {
    #[param(min = 1, max = 100, default = 8)]
    pub block_size: u32,
}

impl Filter for Pixelate {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let bs = self.block_size.max(1) as usize;
        let mut out = input.to_vec();

        let mut by = 0;
        while by < h {
            let bh = bs.min(h - by);
            let mut bx = 0;
            while bx < w {
                let bw = bs.min(w - bx);
                let count = (bw * bh) as f32;

                // Compute mean color
                let mut sum = [0.0f32; 3];
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((by + dy) * w + (bx + dx)) * 4;
                        sum[0] += input[idx];
                        sum[1] += input[idx + 1];
                        sum[2] += input[idx + 2];
                    }
                }
                let mean = [sum[0] / count, sum[1] / count, sum[2] / count];

                // Fill block
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((by + dy) * w + (bx + dx)) * 4;
                        out[idx] = mean[0];
                        out[idx + 1] = mean[1];
                        out[idx + 2] = mean[2];
                    }
                }

                bx += bs;
            }
            by += bs;
        }
        Ok(out)
    }
}

impl GpuFilter for Pixelate {
    fn shader_body(&self) -> &str {
        include_str!("../../shaders/pixelate.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, self.block_size.max(1));
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}
