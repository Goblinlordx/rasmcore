use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::Filter;

/// Gaussian vignette — elliptical darkening with Gaussian blur transition.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "vignette", category = "enhancement", cost = "O(n)")]
pub struct Vignette {
    #[param(min = 0.0, max = 100.0, default = 10.0)]
    pub sigma: f32,
    #[param(min = 0, max = 1000, default = 0)]
    pub x_inset: u32,
    #[param(min = 0, max = 1000, default = 0)]
    pub y_inset: u32,
}

impl Filter for Vignette {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;

        // Build elliptical mask
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let rx = (cx - self.x_inset as f32).max(1.0);
        let ry = (cy - self.y_inset as f32).max(1.0);

        let mut mask = vec![1.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f32 - cx) / rx;
                let dy = (y as f32 - cy) / ry;
                let dist2 = dx * dx + dy * dy;
                if dist2 > 1.0 {
                    mask[y * w + x] = 0.0;
                }
            }
        }

        // Blur the mask for smooth transition
        if self.sigma > 0.0 {
            let blur = GaussianBlur { radius: self.sigma };
            // Pack mask as RGBA for blur
            let mut mask_rgba: Vec<f32> = mask.iter().flat_map(|&v| [v, v, v, 1.0]).collect();
            mask_rgba = blur.compute(&mask_rgba, width, height)?;
            for (i, pixel) in mask_rgba.chunks_exact(4).enumerate() {
                mask[i] = pixel[0];
            }
        }

        // Apply mask
        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let m = mask[y * w + x];
                out[idx] *= m;
                out[idx + 1] *= m;
                out[idx + 2] *= m;
            }
        }

        Ok(out)
    }
}

// ── Vignette (Gaussian) GPU — single-pass analytical falloff ────────────

gpu_filter!(Vignette,
    shader: crate::gpu_shaders::vignette::VIGNETTE_GAUSSIAN,
    workgroup: [16, 16, 1],
    params(self_, w, h) => [
        w, h,
        self_.sigma,
        self_.x_inset as f32,
        self_.y_inset as f32,
        0u32, 0u32, 0u32
    ]
);
