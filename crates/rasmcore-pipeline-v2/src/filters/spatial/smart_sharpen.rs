use crate::node::{GpuShader, PipelineError};
use crate::ops::{Filter, GpuFilter};

use super::bilateral::Bilateral;
use crate::gpu_shaders::spatial;

/// Smart sharpen — edge-preserving sharpening via bilateral unsharp mask.
///
/// Uses a bilateral filter (instead of Gaussian) for the blur step,
/// preserving edges while sharpening. Formula:
/// `output = input + amount * (input - bilateral_blur(input))`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "smart_sharpen",
    category = "spatial",
    cost = "O(n * radius^2) via bilateral"
)]
pub struct SmartSharpen {
    /// Sharpening strength.
    #[param(min = 0.0, max = 5.0, default = 1.0)]
    pub amount: f32,
    /// Bilateral filter radius.
    #[param(min = 1, max = 20, default = 3)]
    pub radius: u32,
    /// Edge preservation threshold (bilateral sigma_color).
    #[param(min = 0.01, max = 1.0, default = 0.1)]
    pub threshold: f32,
}

impl Filter for SmartSharpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.amount.abs() < 1e-6 {
            return Ok(input.to_vec());
        }
        // Edge-preserving blur via bilateral
        let bilateral = Bilateral {
            diameter: self.radius * 2 + 1,
            sigma_color: self.threshold,
            sigma_space: self.radius as f32,
        };
        let blurred = bilateral.compute(input, width, height)?;

        // Unsharp mask: output = input + amount * (input - blurred)
        let amount = self.amount;
        let mut out = input.to_vec();
        for (i, pixel) in out.chunks_exact_mut(4).enumerate() {
            let bi = i * 4;
            pixel[0] += amount * (pixel[0] - blurred[bi]);
            pixel[1] += amount * (pixel[1] - blurred[bi + 1]);
            pixel[2] += amount * (pixel[2] - blurred[bi + 2]);
            // alpha unchanged
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 {
        self.radius + 4 // bilateral radius + safety margin
    }
}

// ── SmartSharpen GPU (bilateral blur + unsharp mask) ────────────────────────
// Uses bilateral shader for blur, then sharpen_apply for unsharp mask.

impl GpuFilter for SmartSharpen {
    fn shader_body(&self) -> &str {
        ""
    } // multi-pass only
    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }
    fn params(&self, _w: u32, _h: u32) -> Vec<u8> {
        Vec::new()
    }

    fn gpu_shaders(&self, w: u32, h: u32) -> Vec<GpuShader> {
        if self.amount.abs() < 1e-6 {
            return vec![self.gpu_shader(w, h)];
        }
        let r = self.radius;
        let sc2 = -0.5f32 / (self.threshold * self.threshold);
        let ss2 = -0.5f32 / (r as f32 * r as f32);
        let n = w * h;

        // Pass 1: bilateral blur
        let bilateral_params = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&w.to_le_bytes());
            buf.extend_from_slice(&h.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&sc2.to_le_bytes());
            buf.extend_from_slice(&ss2.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };
        // Pass 2: unsharp mask
        let sharpen_params = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&w.to_le_bytes());
            buf.extend_from_slice(&h.to_le_bytes());
            buf.extend_from_slice(&self.amount.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };

        vec![
            GpuShader {
                body: spatial::BILATERAL.to_string(),
                entry_point: "main",
                workgroup_size: [16, 16, 1],
                params: bilateral_params,
                extra_buffers: vec![],
                reduction_buffers: vec![],
                convergence_check: None,
                loop_dispatch: None,
                setup: None,
            },
            GpuShader {
                body: spatial::SHARPEN_APPLY.to_string(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: sharpen_params,
                extra_buffers: vec![vec![0u8; (n * 16) as usize]], // placeholder for original
                reduction_buffers: vec![],
                convergence_check: None,
                loop_dispatch: None,
                setup: None,
            },
        ]
    }
}
