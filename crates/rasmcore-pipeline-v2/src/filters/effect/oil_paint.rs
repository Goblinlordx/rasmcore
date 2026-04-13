use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::gpu_params_push_u32;

/// Oil paint — neighborhood mode filter matching ImageMagick's `-paint`.
///
/// For each pixel: bin neighborhood by sRGB-encoded luminance intensity
/// (256 bins), find the mode bin, copy the pixel that first raised the
/// mode count. No averaging — copies a single representative pixel.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "oil_paint", category = "effect", cost = "O(n * radius^2)")]
pub struct OilPaint {
    #[param(min = 1, max = 20, default = 4)]
    pub radius: u32,
}

impl Filter for OilPaint {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as usize;
        const BINS: usize = 256;
        let mut out = input.to_vec();

        let kw = 2 * r + 1; // kernel width = 2*ceil(radius)+1 (matches IM)
        for y in 0..h {
            for x in 0..w {
                let mut count = [0u32; BINS];
                let mut max_count = 0u32;
                let mut best_pixel = (y * w + x) * 4; // default: self

                // Iterate full kernel, clamp out-of-bounds to edge (matching IM virtual pixels).
                // Edge pixels get counted multiple times when the kernel extends past the border.
                for v in 0..kw {
                    for u in 0..kw {
                        let ny = (y as i64 + v as i64 - r as i64).clamp(0, h as i64 - 1) as usize;
                        let nx = (x as i64 + u as i64 - r as i64).clamp(0, w as i64 - 1) as usize;
                        let idx = (ny * w + nx) * 4;
                        // IM Rec709Luma: direct weighted sum (no gamma encode since
                        // IM sees our TIFF as sRGB colorspace, not linear RGB)
                        let intensity = 0.212656 * input[idx]
                            + 0.715158 * input[idx + 1]
                            + 0.072186 * input[idx + 2];
                        // Round-to-nearest (matching IM's ScaleQuantumToChar: +0.5)
                        let bin = ((intensity.max(0.0) * 255.0 + 0.5) as usize).min(BINS - 1);
                        count[bin] += 1;
                        // First pixel to reach a new max wins (IM: strict >)
                        if count[bin] > max_count {
                            max_count = count[bin];
                            best_pixel = idx;
                        }
                    }
                }

                let oidx = (y * w + x) * 4;
                out[oidx] = input[best_pixel];
                out[oidx + 1] = input[best_pixel + 1];
                out[oidx + 2] = input[best_pixel + 2];
            }
        }
        Ok(out)
    }
}

impl GpuFilter for OilPaint {
    fn shader_body(&self) -> &str {
        include_str!("../../shaders/oil_paint.wgsl")
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [8, 8, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, self.radius);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}
