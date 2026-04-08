//! PerspectiveCorrect filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::PerspectiveWarp;

// Perspective Correct — keystone correction (simplified)
// ═══════════════════════════════════════════════════════════════════════════

/// Keystone correction — correct converging vertical/horizontal lines.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "perspective_correct", category = "transform")]
pub struct PerspectiveCorrect {
    #[param(min = -0.005, max = 0.005, step = 0.0001, default = 0.0)] pub vertical: f32,
    #[param(min = -0.005, max = 0.005, step = 0.0001, default = 0.0)] pub horizontal: f32,
}

impl Filter for PerspectiveCorrect {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        // Delegate to PerspectiveWarp with appropriate homography
        let warp = PerspectiveWarp {
            h11: 1.0, h12: 0.0, h13: 0.0,
            h21: 0.0, h22: 1.0, h23: 0.0,
            h31: self.horizontal, h32: self.vertical,
        };
        warp.compute(input, width, height)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let warp = PerspectiveWarp {
            h11: 1.0, h12: 0.0, h13: 0.0,
            h21: 0.0, h22: 1.0, h23: 0.0,
            h31: self.horizontal, h32: self.vertical,
        };
        warp.gpu_shader_passes(width, height)
    }
}
