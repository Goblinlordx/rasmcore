//! Dilate morphology filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::{dilate_cpu, make_dilate_shader};

// Dilate
// ═══════════════════════════════════════════════════════════════════════════

/// Dilate — maximum filter (expands bright regions).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "dilate", category = "morphology")]
pub struct Dilate {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for Dilate {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(dilate_cpu(input, width, height, self.radius))
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        Some(vec![make_dilate_shader(width, height, self.radius)])
    }

    fn tile_overlap(&self) -> u32 { self.radius }
}
