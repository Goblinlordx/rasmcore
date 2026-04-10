use crate::node::PipelineError;
use crate::ops::Filter;

use super::{accum4_unit, clamp_coord};
use crate::gpu_shaders::spatial;

/// Motion blur — linear directional blur.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "motion_blur", category = "spatial", cost = "O(n * length)")]
pub struct MotionBlur {
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle: f32, // degrees
    #[param(min = 0.0, max = 200.0, default = 10.0)]
    pub length: f32, // pixels
}

impl Filter for MotionBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.length <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let rad = self.angle.to_radians();
        let dx = rad.cos();
        let dy = rad.sin();
        let steps = self.length.ceil() as usize;
        let inv_steps = 1.0 / (steps as f32 + 1.0);
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for s in 0..=steps {
                    let t = s as f32 - steps as f32 * 0.5;
                    let sx = clamp_coord((x as f32 + t * dx).round() as i32, w);
                    let sy = clamp_coord((y as f32 + t * dy).round() as i32, h);
                    let idx = (sy * w + sx) * 4;
                    accum4_unit(&mut sum, &input[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                out[out_idx] = sum[0] * inv_steps;
                out[out_idx + 1] = sum[1] * inv_steps;
                out[out_idx + 2] = sum[2] * inv_steps;
                out[out_idx + 3] = sum[3] * inv_steps;
            }
        }

        Ok(out)
    }
}

// ── MotionBlur GPU (single-pass directional) ────────────────────────────────

gpu_filter!(MotionBlur,
    shader: spatial::MOTION_BLUR,
    workgroup: [256, 1, 1],
    params(self_, w, h) => [
        w, h, self_.length.ceil() as u32, 0u32,
        self_.angle.to_radians().cos(),
        self_.angle.to_radians().sin(),
        1.0f32 / (self_.length.ceil() + 1.0),
        0u32
    ]
);
