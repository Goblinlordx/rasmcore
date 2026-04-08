//! MorphBlackhat morphology filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::{dilate_cpu, erode_cpu, make_dilate_shader, make_erode_shader, make_snapshot_shader, make_sub_shader, SUB_CURRENT_MINUS_SNAP_WGSL};

// Morph Black Hat (close − input)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological black hat — closing minus input (isolates dark details).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_blackhat", category = "morphology")]
pub struct MorphBlackhat {
    #[param(min = 1, max = 20, step = 1, default = 2, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphBlackhat {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let dilated = dilate_cpu(input, width, height, self.radius);
        let closed = erode_cpu(&dilated, width, height, self.radius);
        let mut out = vec![0.0f32; input.len()];
        for i in (0..input.len()).step_by(4) {
            out[i] = (closed[i] - input[i]).max(0.0);
            out[i + 1] = (closed[i + 1] - input[i + 1]).max(0.0);
            out[i + 2] = (closed[i + 2] - input[i + 2]).max(0.0);
            out[i + 3] = input[i + 3];
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        // Pass 0: snapshot original → reduction buffer, passthrough to output
        // Pass 1: dilate
        // Pass 2: erode (close result)
        // Pass 3: output = close − snapshot (closed − original)
        Some(vec![
            make_snapshot_shader(width, height, 0),
            make_dilate_shader(width, height, self.radius),
            make_erode_shader(width, height, self.radius),
            make_sub_shader(SUB_CURRENT_MINUS_SNAP_WGSL, width, height, 0),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius * 2 }
}
