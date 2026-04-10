//! Erode morphology filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::{erode_cpu, make_erode_shader};

// Erode
// ═══════════════════════════════════════════════════════════════════════════

/// Erode — minimum filter (expands dark regions).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "erode", category = "morphology")]
pub struct Erode {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for Erode {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(erode_cpu(input, width, height, self.radius))
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        Some(vec![make_erode_shader(width, height, self.radius)])
    }

    fn tile_overlap(&self) -> u32 {
        self.radius
    }
}
