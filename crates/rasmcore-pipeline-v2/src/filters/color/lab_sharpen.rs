use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::{rgb_to_lab, lab_to_rgb, gaussian_blur_1d};

/// Lab sharpen — unsharp mask on L channel only (preserves chrominance).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "lab_sharpen", category = "color", cost = "O(n * radius) via gaussian_blur")]
pub struct LabSharpen {
    /// Sharpening strength (0-10).
    #[param(min = 0.0, max = 10.0, default = 1.0)]
    pub amount: f32,
    /// Blur radius for unsharp mask.
    #[param(min = 0.0, max = 100.0, default = 2.0)]
    pub radius: f32,
}

impl Filter for LabSharpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let pixel_count = w * h;
        // Convert to Lab
        let mut l_channel = vec![0.0f32; pixel_count];
        let mut a_channel = vec![0.0f32; pixel_count];
        let mut b_channel = vec![0.0f32; pixel_count];
        for i in 0..pixel_count {
            let idx = i * 4;
            let (l, a, b) = rgb_to_lab(input[idx], input[idx + 1], input[idx + 2]);
            l_channel[i] = l;
            a_channel[i] = a;
            b_channel[i] = b;
        }
        // Gaussian blur the L channel
        let blurred_l = gaussian_blur_1d(&l_channel, w, h, self.radius);
        // Unsharp mask: sharpened = original + amount * (original - blurred)
        for i in 0..pixel_count {
            l_channel[i] += self.amount * (l_channel[i] - blurred_l[i]);
            l_channel[i] = l_channel[i].clamp(0.0, 100.0);
        }
        // Convert back to RGB
        let mut out = input.to_vec();
        for i in 0..pixel_count {
            let idx = i * 4;
            let (r, g, b) = lab_to_rgb(l_channel[i], a_channel[i], b_channel[i]);
            out[idx] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
        }
        Ok(out)
    }
}

impl GpuFilter for LabSharpen {
    fn shader_body(&self) -> &str {
        include_str!("../../shaders/lab_sharpen.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.amount.to_le_bytes());
        buf.extend_from_slice(&(self.radius.ceil() as u32).to_le_bytes());
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_sharpen_preserves_flat() {
        // Flat gray image — sharpening should have no effect
        let input: Vec<f32> = (0..16).flat_map(|_| vec![0.5, 0.5, 0.5, 1.0]).collect();
        let f = LabSharpen {
            amount: 2.0,
            radius: 1.0,
        };
        let out = f.compute(&input, 4, 4).unwrap();
        for pixel in out.chunks_exact(4) {
            assert!(
                (pixel[0] - 0.5).abs() < 0.02
                    && (pixel[1] - 0.5).abs() < 0.02
                    && (pixel[2] - 0.5).abs() < 0.02,
                "Flat image should be unchanged after sharpening"
            );
        }
    }
}
