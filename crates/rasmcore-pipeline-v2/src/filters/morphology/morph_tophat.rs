//! MorphTophat morphology filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::{dilate_cpu, erode_cpu, make_dilate_shader, make_erode_shader, make_snapshot_shader, make_sub_shader, SUB_SNAP_MINUS_CURRENT_WGSL};

// Morph Top Hat (input − open)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological top hat — input minus opening (isolates bright details).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_tophat", category = "morphology")]
pub struct MorphTophat {
    #[param(min = 1, max = 20, step = 1, default = 2, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphTophat {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let eroded = erode_cpu(input, width, height, self.radius);
        let opened = dilate_cpu(&eroded, width, height, self.radius);
        let mut out = vec![0.0f32; input.len()];
        for i in (0..input.len()).step_by(4) {
            out[i] = (input[i] - opened[i]).max(0.0);
            out[i + 1] = (input[i + 1] - opened[i + 1]).max(0.0);
            out[i + 2] = (input[i + 2] - opened[i + 2]).max(0.0);
            out[i + 3] = input[i + 3];
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        // Pass 0: snapshot original → reduction buffer, passthrough to output
        // Pass 1: erode
        // Pass 2: dilate (open result)
        // Pass 3: output = snapshot − open (original − opened)
        Some(vec![
            make_snapshot_shader(width, height, 0),
            make_erode_shader(width, height, self.radius),
            make_dilate_shader(width, height, self.radius),
            make_sub_shader(SUB_SNAP_MINUS_CURRENT_WGSL, width, height, 0),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius * 2 }
}
