//! MorphOpen morphology filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::{dilate_cpu, erode_cpu, make_dilate_shader, make_erode_shader};

// Morph Open (erode → dilate)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological opening — erode then dilate (removes small bright spots).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_open", category = "morphology")]
pub struct MorphOpen {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphOpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let eroded = erode_cpu(input, width, height, self.radius);
        Ok(dilate_cpu(&eroded, width, height, self.radius))
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        Some(vec![
            make_erode_shader(width, height, self.radius),
            make_dilate_shader(width, height, self.radius),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius * 2 }
}
