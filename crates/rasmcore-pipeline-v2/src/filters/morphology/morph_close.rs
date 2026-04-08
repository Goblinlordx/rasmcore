//! MorphClose morphology filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::{dilate_cpu, erode_cpu, make_dilate_shader, make_erode_shader};

// Morph Close (dilate → erode)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological closing — dilate then erode (fills small dark holes).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_close", category = "morphology")]
pub struct MorphClose {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphClose {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let dilated = dilate_cpu(input, width, height, self.radius);
        Ok(erode_cpu(&dilated, width, height, self.radius))
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        Some(vec![
            make_dilate_shader(width, height, self.radius),
            make_erode_shader(width, height, self.radius),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius * 2 }
}
