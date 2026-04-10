//! Checkerboard generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Checkerboard
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a checkerboard pattern.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "checkerboard", category = "generator")]
pub struct Checkerboard {
    #[param(min = 2.0, max = 500.0, step = 1.0, default = 32.0)]
    pub size: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color1_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color1_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color1_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)]
    pub color2_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)]
    pub color2_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)]
    pub color2_b: f32,
}

const CHECKERBOARD_WGSL: &str = r#"
struct Params { width: u32, height: u32, size: f32, c1r: f32, c1g: f32, c1b: f32, c2r: f32, c2g: f32, c2b: f32, _pad: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let cx = u32(floor(x / params.size)); let cy = u32(floor(y / params.size));
  let check = (cx + cy) % 2u;
  var color: vec3<f32>;
  if (check == 0u) { color = vec3<f32>(params.c1r, params.c1g, params.c1b); }
  else { color = vec3<f32>(params.c2r, params.c2g, params.c2b); }
  output[idx] = vec4<f32>(color, 1.0);
}
"#;

impl Filter for Checkerboard {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let cx = (x as f32 / self.size).floor() as u32;
                let cy = (y as f32 / self.size).floor() as u32;
                let i = ((y * width + x) * 4) as usize;
                if (cx + cy) % 2 == 0 {
                    out[i] = self.color1_r;
                    out[i + 1] = self.color1_g;
                    out[i + 2] = self.color1_b;
                } else {
                    out[i] = self.color2_r;
                    out[i + 1] = self.color2_g;
                    out[i + 2] = self.color2_b;
                }
                out[i + 3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.size);
        gpu_push_f32(&mut p, self.color1_r);
        gpu_push_f32(&mut p, self.color1_g);
        gpu_push_f32(&mut p, self.color1_b);
        gpu_push_f32(&mut p, self.color2_r);
        gpu_push_f32(&mut p, self.color2_g);
        gpu_push_f32(&mut p, self.color2_b);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            CHECKERBOARD_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
