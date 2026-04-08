use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{gpu_params_push_f32, gpu_params_push_u32};

/// Halftone — CMYK-style dot pattern via sine-wave screening.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "halftone", category = "effect", cost = "O(n)")]
pub struct Halftone {
    #[param(min = 1.0, max = 50.0, default = 8.0)]
    pub dot_size: f32,
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle_offset: f32,
}

impl Filter for Halftone {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let ds = self.dot_size.max(1.0);
        let freq = std::f32::consts::PI / ds;

        // CMYK screen angles
        let angles = [
            (15.0 + self.angle_offset).to_radians(),
            (75.0 + self.angle_offset).to_radians(),
            (0.0 + self.angle_offset).to_radians(),
            (45.0 + self.angle_offset).to_radians(),
        ];

        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let r = input[idx];
                let g = input[idx + 1];
                let b = input[idx + 2];

                // Simple RGB to CMYK
                let k = 1.0 - r.max(g).max(b);
                let inv_k = if k < 1.0 { 1.0 / (1.0 - k) } else { 0.0 };
                let c = (1.0 - r - k) * inv_k;
                let m = (1.0 - g - k) * inv_k;
                let yc = (1.0 - b - k) * inv_k;

                let cmyk = [c, m, yc, k];
                let mut screened = [0.0f32; 4];

                for (i, &angle) in angles.iter().enumerate() {
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let rx = x as f32 * cos_a + y as f32 * sin_a;
                    let ry = -(x as f32) * sin_a + y as f32 * cos_a;
                    let screen = (rx * freq).sin() * (ry * freq).sin();
                    let threshold = (screen + 1.0) * 0.5;
                    screened[i] = if cmyk[i] > threshold { 1.0 } else { 0.0 };
                }

                // CMYK back to RGB
                let inv_k2 = 1.0 - screened[3];
                out[idx] = (1.0 - screened[0]) * inv_k2;
                out[idx + 1] = (1.0 - screened[1]) * inv_k2;
                out[idx + 2] = (1.0 - screened[2]) * inv_k2;
            }
        }
        Ok(out)
    }
}

impl GpuFilter for Halftone {
    fn shader_body(&self) -> &str {
        include_str!("../../shaders/halftone.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let ds = self.dot_size.max(1.0);
        let freq = std::f32::consts::PI / ds;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, freq);
        gpu_params_push_f32(&mut buf, (15.0 + self.angle_offset).to_radians()); // C angle
        gpu_params_push_f32(&mut buf, (75.0 + self.angle_offset).to_radians()); // M angle
        gpu_params_push_f32(&mut buf, (0.0 + self.angle_offset).to_radians());  // Y angle
        gpu_params_push_f32(&mut buf, (45.0 + self.angle_offset).to_radians()); // K angle
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}
