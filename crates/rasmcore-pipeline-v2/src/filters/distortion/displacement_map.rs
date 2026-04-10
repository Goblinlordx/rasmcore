//! DisplacementMap distortion filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_f32, gpu_params_push_u32};

// Displacement Map
// ═══════════════════════════════════════════════════════════════════════════

/// Displacement map — uses the image's own color channels as displacement.
/// Red channel displaces X, green channel displaces Y.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "displacement_map", category = "distortion")]
pub struct DisplacementMap {
    #[param(min = 0.0, max = 100.0, step = 0.5, default = 10.0)]
    pub scale: f32,
}

const DISPLACEMENT_MAP_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];
  let dx = (px.r - 0.5) * params.scale;
  let dy = (px.g - 0.5) * params.scale;
  let sx = f32(x) + dx; let sy = f32(y) + dy;
  output[idx] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for DisplacementMap {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) * 4) as usize;
                let dx = (input[i] - 0.5) * self.scale;
                let dy = (input[i + 1] - 0.5) * self.scale;
                let sx = x as f32 + dx;
                let sy = y as f32 + dy;
                let px = sample_bilinear(input, width, height, sx, sy);
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{DISPLACEMENT_MAP_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.scale);
        gpu_params_push_u32(&mut params, 0); // pad
        Some(vec![crate::node::GpuShader::new(
            shader,
            "main",
            [16, 16, 1],
            params,
        )])
    }

    fn tile_overlap(&self) -> u32 {
        0
    }
}
