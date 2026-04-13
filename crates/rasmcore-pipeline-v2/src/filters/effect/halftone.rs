use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::gpu_params_push_f32;

/// Halftone — Photoshop-style CMYK circular dot screening.
///
/// Matches Photoshop's "Color Halftone" (Filter → Pixelate) algorithm:
/// 1. Decompose RGB → CMYK
/// 2. Per CMYK channel: rotate coordinate grid by screen angle, find cell center,
///    sample ink density, compute dot radius = max_radius × sqrt(density)
/// 3. Point-in-circle test with anti-aliased edge
/// 4. Subtractive composite CMYK → RGB
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "halftone", category = "effect", cost = "O(n)")]
pub struct Halftone {
    /// Max dot radius in pixels (also half the cell spacing).
    #[param(min = 1.0, max = 50.0, default = 8.0)]
    pub dot_size: f32,
    /// Rotation offset added to all screen angles (degrees).
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle_offset: f32,
}

impl Filter for Halftone {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let max_r = self.dot_size.max(1.0);
        let cell = 2.0 * max_r; // cell size = 2 × max_radius

        // CMYK screen angles (industry standard + user offset)
        let angles = [
            (15.0 + self.angle_offset).to_radians(),  // Cyan
            (75.0 + self.angle_offset).to_radians(),  // Magenta
            (0.0 + self.angle_offset).to_radians(),   // Yellow
            (45.0 + self.angle_offset).to_radians(),  // Black
        ];

        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let mut ink = [0.0f32; 4];

                for (i, &angle) in angles.iter().enumerate() {
                    let (sin_a, cos_a) = angle.sin_cos();

                    // Rotate pixel into screen space
                    let sx = x as f32 * cos_a + y as f32 * sin_a;
                    let sy = -(x as f32) * sin_a + y as f32 * cos_a;

                    // Find nearest cell center in screen space
                    let cx = (sx / cell).round() * cell;
                    let cy = (sy / cell).round() * cell;

                    // Map cell center back to image space for sampling
                    let img_x = cx * cos_a - cy * (-sin_a);
                    let img_y = cx * sin_a + cy * cos_a;

                    // Sample ink density at cell center (clamp to image bounds)
                    let sx_img = img_x.clamp(0.0, (w - 1) as f32) as usize;
                    let sy_img = img_y.clamp(0.0, (h - 1) as f32) as usize;
                    let sample_idx = (sy_img * w + sx_img) * 4;
                    let sr = input[sample_idx];
                    let sg = input[sample_idx + 1];
                    let sb = input[sample_idx + 2];
                    let sk = 1.0 - sr.max(sg).max(sb);
                    let s_inv_k = if sk < 1.0 { 1.0 / (1.0 - sk) } else { 0.0 };
                    let density = match i {
                        0 => (1.0 - sr - sk) * s_inv_k,
                        1 => (1.0 - sg - sk) * s_inv_k,
                        2 => (1.0 - sb - sk) * s_inv_k,
                        _ => sk,
                    };

                    // Dot radius: area-proportional (r = max_r * sqrt(density))
                    let dot_r = max_r * density.max(0.0).sqrt();

                    // Distance from pixel to cell center in screen space
                    let dist = ((sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)).sqrt();

                    // Anti-aliased coverage (smoothstep at dot edge)
                    ink[i] = smoothstep(dot_r + 0.5, dot_r - 0.5, dist);
                }

                // Subtractive CMYK → RGB composite
                out[idx] = (1.0 - ink[0]) * (1.0 - ink[3]);
                out[idx + 1] = (1.0 - ink[1]) * (1.0 - ink[3]);
                out[idx + 2] = (1.0 - ink[2]) * (1.0 - ink[3]);
            }
        }
        Ok(out)
    }
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 >= edge1 {
        return if x <= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    // Inverted: 1 at center, 0 at edge
    1.0 - t * t * (3.0 - 2.0 * t)
}

impl GpuFilter for Halftone {
    fn shader_body(&self) -> &str {
        include_str!("../../shaders/halftone.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let max_r = self.dot_size.max(1.0);
        let cell = 2.0 * max_r;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, max_r);
        gpu_params_push_f32(&mut buf, cell);
        gpu_params_push_f32(&mut buf, (15.0 + self.angle_offset).to_radians());
        gpu_params_push_f32(&mut buf, (75.0 + self.angle_offset).to_radians());
        gpu_params_push_f32(&mut buf, (0.0 + self.angle_offset).to_radians());
        gpu_params_push_f32(&mut buf, (45.0 + self.angle_offset).to_radians());
        buf
    }
}
