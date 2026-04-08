use crate::node::PipelineError;
use crate::ops::Filter;

use super::{accum4_unit, clamp_coord, blur_params};
use crate::gpu_shaders::spatial;
use crate::node::GpuShader;

/// Box blur — running average within radius. O(1) per pixel.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "box_blur", category = "spatial", cost = "O(n * r) separable")]
pub struct BoxBlur {
    #[param(min = 0, max = 100, default = 1)]
    pub radius: u32,
}

impl Filter for BoxBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius == 0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as i32;
        let diameter = (2 * r + 1) as f32;
        let inv_d = 1.0 / diameter;

        // Horizontal pass
        let mut tmp = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for dx in -r..=r {
                    let sx = clamp_coord(x as i32 + dx, w);
                    let idx = (y * w + sx) * 4;
                    accum4_unit(&mut sum, &input[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    tmp[out_idx + c] = sum[c] * inv_d;
                }
            }
        }

        // Vertical pass
        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for dy in -r..=r {
                    let sy = clamp_coord(y as i32 + dy, h);
                    let idx = (sy * w + x) * 4;
                    accum4_unit(&mut sum, &tmp[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    out[out_idx + c] = sum[c] * inv_d;
                }
            }
        }

        Ok(out)
    }

    fn tile_overlap(&self) -> u32 {
        self.radius
    }
}

// ── BoxBlur GPU (2-pass separable H+V) ───────────────────────────────────────

gpu_filter_passes_only!(BoxBlur,
    passes(self_, w, h) => {
        let params = blur_params(w, h, self_.radius);
        vec![
            GpuShader::new(spatial::BOX_BLUR_H.to_string(), "main", [256, 1, 1], params.clone()),
            GpuShader::new(spatial::BOX_BLUR_V.to_string(), "main", [256, 1, 1], params),
        ]
    }
);
