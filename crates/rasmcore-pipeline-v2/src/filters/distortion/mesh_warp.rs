//! MeshWarp distortion filter.

use crate::node::{PipelineError};
use crate::ops::Filter;

use std::f32::consts::PI;
use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_f32};

// Mesh Warp (simplified — uniform grid displacement)
// ═══════════════════════════════════════════════════════════════════════════

/// Mesh warp — grid-based displacement mapping.
/// Uses a uniform displacement field for simplicity (no control point grid).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "mesh_warp", category = "distortion")]
pub struct MeshWarp {
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 10.0)]
    pub strength: f32,
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 40.0)]
    pub frequency: f32,
}

const MESH_WARP_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, strength: f32, frequency: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let dx = params.strength * sin(2.0 * PI * f32(y) / params.frequency);
  let dy = params.strength * sin(2.0 * PI * f32(x) / params.frequency);
  let sx = f32(x) + dx; let sy = f32(y) + dy;
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for MeshWarp {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        for y in 0..height {
            for x in 0..width {
                let dx = self.strength * (2.0 * PI * y as f32 / self.frequency).sin();
                let dy = self.strength * (2.0 * PI * x as f32 / self.frequency).sin();
                let sx = x as f32 + dx;
                let sy = y as f32 + dy;
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{MESH_WARP_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.strength);
        gpu_params_push_f32(&mut params, self.frequency);
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}
