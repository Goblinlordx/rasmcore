//! Barrel distortion filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_f32};

// Barrel Distortion
// ═══════════════════════════════════════════════════════════════════════════

/// Barrel/pincushion distortion.
/// k1 > 0: barrel, k1 < 0: pincushion.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "barrel", category = "distortion")]
pub struct Barrel {
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.3)]
    pub k1: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub k2: f32,
}

const BARREL_WGSL: &str = r#"
struct Params { width: u32, height: u32, k1: f32, k2: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let cx = f32(params.width) * 0.5; let cy = f32(params.height) * 0.5;
  let nx = (f32(x) - cx) / cx; let ny = (f32(y) - cy) / cy;
  let r2 = nx * nx + ny * ny; let r4 = r2 * r2;
  let d = 1.0 + params.k1 * r2 + params.k2 * r4;
  let sx = nx * d * cx + cx; let sy = ny * d * cy + cy;
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Barrel {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        for y in 0..height {
            for x in 0..width {
                let nx = (x as f32 - cx) / cx;
                let ny = (y as f32 - cy) / cy;
                let r2 = nx * nx + ny * ny;
                let r4 = r2 * r2;
                let d = 1.0 + self.k1 * r2 + self.k2 * r4;
                let sx = nx * d * cx + cx;
                let sy = ny * d * cy + cy;
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        None
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{BARREL_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.k1);
        gpu_params_push_f32(&mut params, self.k2);
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
