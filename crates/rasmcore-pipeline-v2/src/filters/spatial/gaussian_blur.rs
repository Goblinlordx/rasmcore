use crate::node::PipelineError;
use crate::ops::Filter;

use super::{accum4, blur_params, clamp_coord, gaussian_kernel_1d, gaussian_kernel_bytes};
use crate::gpu_shaders::spatial;
use crate::node::GpuShader;

/// Gaussian blur — separable convolution on f32 data.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "gaussian_blur",
    category = "spatial",
    cost = "O(n * r) separable",
    doc = "docs/operations/filters/spatial/gaussian_blur.adoc"
)]
pub struct GaussianBlur {
    /// Blur radius in pixels. Larger values produce stronger blur.
    #[param(min = 0.0, max = 100.0, default = 1.0)]
    pub radius: f32,
}

impl Filter for GaussianBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let kernel = gaussian_kernel_1d(self.radius);
        let r = kernel.len() / 2;

        // Pass 1: horizontal — fast interior path skips clamp_coord for most pixels
        let mut tmp = vec![0.0f32; w * h * 4];
        for y in 0..h {
            let row_base = y * w * 4;

            // Boundary pixels (left + right margins)
            for x in 0..r.min(w) {
                let mut sum = [0.0f32; 4];
                for (kx, &kw) in kernel.iter().enumerate() {
                    let sx = clamp_coord(x as i32 + kx as i32 - r as i32, w);
                    let idx = row_base + sx * 4;
                    accum4(&mut sum, &input[idx..], kw);
                }
                let out_idx = row_base + x * 4;
                tmp[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
            for x in (w.saturating_sub(r))..w {
                if x < r {
                    continue;
                } // already handled above
                let mut sum = [0.0f32; 4];
                for (kx, &kw) in kernel.iter().enumerate() {
                    let sx = clamp_coord(x as i32 + kx as i32 - r as i32, w);
                    let idx = row_base + sx * 4;
                    accum4(&mut sum, &input[idx..], kw);
                }
                let out_idx = row_base + x * 4;
                tmp[out_idx..out_idx + 4].copy_from_slice(&sum);
            }

            // Interior pixels — no bounds check needed
            let x_start = r.min(w);
            let x_end = w.saturating_sub(r);
            for x in x_start..x_end {
                let mut sum = [0.0f32; 4];
                // All source positions (x-r..x+r) are in bounds
                let base = row_base + (x - r) * 4;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let idx = base + ki * 4;
                    accum4(&mut sum, &input[idx..], kw);
                }
                let out_idx = row_base + x * 4;
                tmp[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
        }

        // Pass 2: vertical — fast interior path
        let mut out = vec![0.0f32; w * h * 4];
        let stride = w * 4;

        // Boundary rows (top + bottom margins)
        for y in 0..r.min(h) {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for (ky, &kw) in kernel.iter().enumerate() {
                    let sy = clamp_coord(y as i32 + ky as i32 - r as i32, h);
                    let idx = sy * stride + x * 4;
                    accum4(&mut sum, &tmp[idx..], kw);
                }
                let out_idx = y * stride + x * 4;
                out[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
        }
        for y in (h.saturating_sub(r))..h {
            if y < r {
                continue;
            }
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for (ky, &kw) in kernel.iter().enumerate() {
                    let sy = clamp_coord(y as i32 + ky as i32 - r as i32, h);
                    let idx = sy * stride + x * 4;
                    accum4(&mut sum, &tmp[idx..], kw);
                }
                let out_idx = y * stride + x * 4;
                out[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
        }

        // Interior rows — no bounds check needed
        let y_start = r.min(h);
        let y_end = h.saturating_sub(r);
        for y in y_start..y_end {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                let px_offset = x * 4;
                let base_row = (y - r) * stride;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let idx = base_row + ki * stride + px_offset;
                    accum4(&mut sum, &tmp[idx..], kw);
                }
                let out_idx = y * stride + x * 4;
                out[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
        }

        Ok(out)
    }
}

impl GaussianBlur {
    pub fn tile_overlap(&self) -> u32 {
        (self.radius * 3.0).ceil() as u32
    }
}

// ── GaussianBlur GPU (2-pass separable H+V) ──────────────────────────────────

gpu_filter_passes_only!(GaussianBlur,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let params = blur_params(w, h, kr);
        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], params.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], params)
                .with_extra_buffers(vec![kb]),
        ]
    }
);
