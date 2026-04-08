use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{clamp_coord, gpu_params_push_u32};

/// Emboss — 3D relief effect via directional convolution kernel.
///
/// `output = convolve(input, emboss_kernel) + 0.5`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "emboss", category = "effect", cost = "O(n)")]
pub struct Emboss;

impl Filter for Emboss {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        // Standard emboss kernel (top-left to bottom-right)
        let kernel: [f32; 9] = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 3];
                for ky in 0..3i32 {
                    for kx in 0..3i32 {
                        let sx = clamp_coord(x as i32 + kx - 1, w);
                        let sy = clamp_coord(y as i32 + ky - 1, h);
                        let k = kernel[(ky * 3 + kx) as usize];
                        let idx = (sy * w + sx) * 4;
                        for c in 0..3 {
                            sum[c] += k * input[idx + c];
                        }
                    }
                }
                let idx = (y * w + x) * 4;
                for c in 0..3 {
                    out[idx + c] = sum[c] + 0.5; // neutral gray offset
                }
                out[idx + 3] = input[idx + 3]; // alpha
            }
        }
        Ok(out)
    }
}

impl GpuFilter for Emboss {
    fn shader_body(&self) -> &str {
        include_str!("../../shaders/emboss.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, 0); // _pad0
        gpu_params_push_u32(&mut buf, 0); // _pad1
        buf
    }
}
