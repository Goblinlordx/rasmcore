use crate::node::PipelineError;
use crate::ops::Filter;

use super::accum4_unit;
use crate::gpu_shaders::spatial;

/// Zoom blur — radial blur from a center point.
///
/// Samples along radial lines from each pixel toward the center,
/// producing a camera zoom effect.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "zoom_blur", category = "spatial", cost = "O(n * samples)")]
pub struct ZoomBlur {
    /// Center X position (0.0-1.0).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub center_x: f32,
    /// Center Y position (0.0-1.0).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub center_y: f32,
    /// Zoom factor. Larger = more blur.
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.1)]
    pub factor: f32,
}

impl Filter for ZoomBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.factor.abs() < 1e-6 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let wi = w as i32;
        let hi = h as i32;
        let cx = self.center_x * w as f32;
        let cy = self.center_y * h as f32;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let dx = cx - x as f32;
                let dy = cy - y as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                let samples = ((dist * self.factor.abs()).ceil() as usize).clamp(3, 64);
                let inv_n = 1.0 / samples as f32;

                let mut sum = [0.0f32; 4];
                for s in 0..samples {
                    let t = s as f32 * self.factor / samples as f32;
                    let sx = (x as f32 + dx * t).round() as i32;
                    let sy = (y as f32 + dy * t).round() as i32;
                    let sx = sx.clamp(0, wi - 1) as usize;
                    let sy = sy.clamp(0, hi - 1) as usize;
                    let idx = (sy * w + sx) * 4;
                    accum4_unit(&mut sum, &input[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                out[out_idx] = sum[0] * inv_n;
                out[out_idx + 1] = sum[1] * inv_n;
                out[out_idx + 2] = sum[2] * inv_n;
                out[out_idx + 3] = sum[3] * inv_n;
            }
        }
        Ok(out)
    }
}

// ── ZoomBlur GPU (single-pass radial sampling) ──────────────────────────────

gpu_filter!(ZoomBlur,
    shader: spatial::ZOOM_BLUR,
    workgroup: [256, 1, 1],
    params(self_, w, h) => [
        w, h,
        ((self_.factor.abs() * 64.0).ceil() as u32).clamp(8, 128),
        0u32,
        self_.center_x * w as f32,
        self_.center_y * h as f32,
        self_.factor,
        0u32
    ]
);
