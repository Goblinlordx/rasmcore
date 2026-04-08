use crate::node::PipelineError;
use crate::ops::Filter;

use super::accum4_unit;
use crate::gpu_shaders::spatial;

/// Spin blur — rotational blur around a center point.
///
/// Samples along circular arcs around the center, producing a
/// rotational motion effect.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "spin_blur", category = "spatial", cost = "O(n * samples)")]
pub struct SpinBlur {
    /// Center X position (0.0-1.0).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub center_x: f32,
    /// Center Y position (0.0-1.0).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub center_y: f32,
    /// Maximum rotation angle in degrees.
    #[param(min = 0.0, max = 360.0, default = 10.0)]
    pub angle: f32,
}

impl Filter for SpinBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.angle.abs() < 1e-6 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let wi = w as i32;
        let hi = h as i32;
        let cx = self.center_x * w as f32;
        let cy = self.center_y * h as f32;
        let angle_rad = self.angle.to_radians();

        // Sample count based on angle and max radius
        let half_diag = ((w * w + h * h) as f32).sqrt() * 0.5;
        let n = ((2.0 * angle_rad.abs() * half_diag).ceil() as usize + 2) | 1; // ensure odd
        let n = n.clamp(3, 129);
        let inv_n = 1.0 / n as f32;

        // Precompute rotation table
        let half = n / 2;
        let cos_sin: Vec<(f32, f32)> = (0..n)
            .map(|i| {
                let offset = angle_rad * (i as f32 - half as f32) / n as f32;
                (offset.cos(), offset.sin())
            })
            .collect();

        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let mut sum = [0.0f32; 4];
                for &(cos_t, sin_t) in &cos_sin {
                    let sx = (cx + dx * cos_t - dy * sin_t).round() as i32;
                    let sy = (cy + dx * sin_t + dy * cos_t).round() as i32;
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

// ── SpinBlur GPU (single-pass rotational sampling) ──────────────────────────

gpu_filter!(SpinBlur,
    shader: spatial::SPIN_BLUR,
    workgroup: [256, 1, 1],
    params(self_, w, h) => [
        w, h,
        ((self_.angle.to_radians().abs() * 32.0).ceil() as u32).clamp(8, 128),
        0u32,
        self_.center_x * w as f32,
        self_.center_y * h as f32,
        self_.angle.to_radians(),
        0u32
    ]
);
