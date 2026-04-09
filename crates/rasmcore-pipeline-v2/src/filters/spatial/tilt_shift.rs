use crate::node::{GpuShader, PipelineError};
use crate::ops::{Filter, GpuFilter};

use super::gaussian_blur::GaussianBlur;
use super::gaussian_kernel_bytes;
use crate::gpu_shaders::spatial;

/// Tilt-shift — selective focus with gradient blur falloff.
///
/// Simulates miniature/tilt-shift photography by blurring areas outside
/// a focus band and blending with the original via a smooth gradient mask.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "tilt_shift", category = "spatial", cost = "O(n * radius) via gaussian_blur")]
pub struct TiltShift {
    /// Focus band center position (0.0 = top, 1.0 = bottom).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub focus_position: f32,
    /// Focus band size as fraction of image height.
    #[param(min = 0.0, max = 1.0, default = 0.2)]
    pub band_size: f32,
    /// Maximum blur radius for out-of-focus areas.
    #[param(min = 0.0, max = 100.0, default = 8.0)]
    pub blur_radius: f32,
    /// Band rotation angle in degrees.
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle: f32,
}

impl Filter for TiltShift {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.blur_radius <= 0.0 || self.band_size >= 1.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;

        // Generate fully blurred version
        let blur = GaussianBlur { radius: self.blur_radius };
        let blurred = blur.compute(input, width, height)?;

        // Blend original and blurred based on distance from focus band
        let angle_rad = self.angle.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let focus_y = self.focus_position;
        let half_band = self.band_size * 0.5;
        let transition = half_band.max(0.05); // transition zone

        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let nx = x as f32 / w as f32 - 0.5;
                let ny = y as f32 / h as f32 - focus_y;
                // Rotate to align with band angle
                let dist = (nx * sin_a + ny * cos_a).abs();
                // Compute mask: 0 inside band, 1 far outside
                let t = if dist < half_band {
                    0.0
                } else {
                    let d = (dist - half_band) / transition;
                    let d = d.min(1.0);
                    d * d * (3.0 - 2.0 * d) // smoothstep
                };
                let idx = (y * w + x) * 4;
                for c in 0..4 {
                    out[idx + c] = input[idx + c] + t * (blurred[idx + c] - input[idx + c]);
                }
            }
        }
        Ok(out)
    }
}

// ── TiltShift GPU (blur pass + blend pass) ──────────────────────────────────

impl GpuFilter for TiltShift {
    fn shader_body(&self) -> &str { "" } // multi-pass only
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, _w: u32, _h: u32) -> Vec<u8> { Vec::new() }

    fn gpu_shaders(&self, w: u32, h: u32) -> Vec<GpuShader> {
        if self.blur_radius <= 0.0 || self.band_size >= 1.0 {
            return vec![self.gpu_shader(w, h)];
        }
        let (krad, kernel_bytes) = gaussian_kernel_bytes(self.blur_radius);
        let n = w * h;

        // Pass 1: horizontal blur
        let h_params = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&w.to_le_bytes());
            buf.extend_from_slice(&h.to_le_bytes());
            buf.extend_from_slice(&krad.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };
        // Pass 2: vertical blur
        let v_params = h_params.clone();
        // Pass 3: blend with mask
        let angle_rad = self.angle.to_radians();
        let half_band = self.band_size * 0.5;
        let transition = half_band.max(0.05);
        let blend_params = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&w.to_le_bytes());
            buf.extend_from_slice(&h.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&self.focus_position.to_le_bytes());
            buf.extend_from_slice(&half_band.to_le_bytes());
            buf.extend_from_slice(&transition.to_le_bytes());
            buf.extend_from_slice(&angle_rad.cos().to_le_bytes());
            buf.extend_from_slice(&angle_rad.sin().to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };

        vec![
            GpuShader {
                body: spatial::GAUSSIAN_BLUR_H.to_string(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: h_params,
                extra_buffers: vec![kernel_bytes.clone()],
                reduction_buffers: vec![],
                convergence_check: None, loop_dispatch: None, setup: None,
            },
            GpuShader {
                body: spatial::GAUSSIAN_BLUR_V.to_string(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: v_params,
                extra_buffers: vec![kernel_bytes],
                reduction_buffers: vec![],
                convergence_check: None, loop_dispatch: None, setup: None,
            },
            GpuShader {
                body: spatial::TILT_SHIFT_BLEND.to_string(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: blend_params,
                extra_buffers: vec![vec![0u8; (n * 16) as usize]], // placeholder for original (snapshot)
                reduction_buffers: vec![],
                convergence_check: None, loop_dispatch: None, setup: None,
            },
        ]
    }
}
