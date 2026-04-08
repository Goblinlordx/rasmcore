//! MorphGradient morphology filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::{dilate_cpu, erode_cpu, make_dilate_shader, make_snapshot_shader, make_sub_shader, make_erode_from_snap_shader, SUB_SNAP_MINUS_CURRENT_WGSL, SNAPSHOT_WGSL};
use super::super::helpers::gpu_params_wh;
use super::gpu_params_push_u32;
use crate::node::ReductionBuffer;

// Morph Gradient (dilate − erode)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological gradient — dilate minus erode (edge detection).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_gradient", category = "morphology")]
pub struct MorphGradient {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphGradient {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let dilated = dilate_cpu(input, width, height, self.radius);
        let eroded = erode_cpu(input, width, height, self.radius);
        let mut out = vec![0.0f32; input.len()];
        for i in (0..input.len()).step_by(4) {
            out[i] = (dilated[i] - eroded[i]).abs();
            out[i + 1] = (dilated[i + 1] - eroded[i + 1]).abs();
            out[i + 2] = (dilated[i + 2] - eroded[i + 2]).abs();
            out[i + 3] = input[i + 3];
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let buf_size = (width as usize) * (height as usize) * 16;
        // Pass 0: snapshot original → buffer 0
        // Pass 1: dilate
        // Pass 2: snapshot dilated → buffer 1
        // Pass 3: erode from buffer 0 (erode the original, not the dilated)
        // Pass 4: diff(buffer 1 [dilate], buffer 0... wait)
        //
        // Actually simpler: after pass 3, erode result is in ping-pong.
        // We need |dilate − erode|. Dilate is in buffer 1. Erode is current input.
        // Use sub shader: |buffer1 − current|
        Some(vec![
            // Pass 0: snapshot original → buf 0
            make_snapshot_shader(width, height, 0),
            // Pass 1: dilate (from original via passthrough)
            make_dilate_shader(width, height, self.radius),
            // Pass 2: snapshot dilate result → buf 1, passthrough
            {
                let mut params = gpu_params_wh(width, height);
                gpu_params_push_u32(&mut params, 0);
                gpu_params_push_u32(&mut params, 0);
                GpuShader {
                    body: SNAPSHOT_WGSL.to_string(),
                    entry_point: "main",
                    workgroup_size: [16, 16, 1],
                    params,
                    extra_buffers: vec![],
                    reduction_buffers: vec![ReductionBuffer {
                        id: 1,
                        initial_data: vec![0u8; buf_size],
                        read_write: true,
                    }],
            convergence_check: None,
            loop_dispatch: None,
                }
            },
            // Pass 3: erode from original (buf 0)
            make_erode_from_snap_shader(width, height, self.radius, 0),
            // Pass 4: |dilate (buf 1) − erode (current ping-pong input)|
            // Uses SUB_SNAP_MINUS_CURRENT: |snapshot − input| where snapshot=buf1=dilate
            make_sub_shader(SUB_SNAP_MINUS_CURRENT_WGSL, width, height, 1),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius }
}
