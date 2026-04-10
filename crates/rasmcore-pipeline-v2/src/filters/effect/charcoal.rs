use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{clamp_coord, gpu_params_push_u32, luminance};

/// Charcoal — edge detection → blur → invert for pencil sketch effect.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "charcoal",
    category = "effect",
    cost = "O(n * radius) via gaussian_blur"
)]
pub struct Charcoal {
    #[param(min = 0.0, max = 20.0, default = 1.0)]
    pub radius: f32,
    #[param(min = 0.0, max = 20.0, default = 1.0)]
    pub sigma: f32,
}

impl Filter for Charcoal {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;

        // Step 1: Sobel edge detection (on luminance)
        let sobel_x: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let mut edges = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;
                for ky in 0..3i32 {
                    for kx in 0..3i32 {
                        let sx = clamp_coord(x as i32 + kx - 1, w);
                        let sy = clamp_coord(y as i32 + ky - 1, h);
                        let idx = (sy * w + sx) * 4;
                        let luma = luminance(input[idx], input[idx + 1], input[idx + 2]);
                        let ki = (ky * 3 + kx) as usize;
                        gx += sobel_x[ki] * luma;
                        gy += sobel_y[ki] * luma;
                    }
                }
                let mag = (gx * gx + gy * gy).sqrt().min(1.0);
                let idx = (y * w + x) * 4;
                edges[idx] = mag;
                edges[idx + 1] = mag;
                edges[idx + 2] = mag;
                edges[idx + 3] = input[idx + 3];
            }
        }

        // Step 2: Blur edges
        let blur = GaussianBlur { radius: self.sigma };
        let blurred = blur.compute(&edges, width, height)?;

        // Step 3: Invert for charcoal look (dark lines on white)
        let mut out = Vec::with_capacity(blurred.len());
        for (i, &v) in blurred.iter().enumerate() {
            if i % 4 == 3 {
                out.push(v); // alpha
            } else {
                out.push(1.0 - v);
            }
        }
        Ok(out)
    }
}

pub(crate) const CHARCOAL_WGSL: &str = include_str!("../../shaders/charcoal.wgsl");

impl GpuFilter for Charcoal {
    fn shader_body(&self) -> &str {
        CHARCOAL_WGSL
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}
