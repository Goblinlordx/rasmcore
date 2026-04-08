//! CloneStamp tool filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, smoothstep_f32};

// Clone Stamp — copy from source offset within circular brush
// ═══════════════════════════════════════════════════════════════════════════

/// Clone stamp — copy pixels from source offset within a circular brush.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "clone_stamp", category = "tool")]
pub struct CloneStamp {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_y: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.1)] pub offset_x: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)] pub offset_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0, hint = "rc.pixels")] pub radius: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub opacity: f32,
}

const CLONE_STAMP_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, ox: f32, oy: f32, radius: f32, opacity: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let dx = x - params.cx; let dy = y - params.cy;
  let dist = sqrt(dx*dx + dy*dy);
  if (dist >= params.radius) { output[idx] = input[idx]; return; }
  let falloff = 1.0 - smoothstep(params.radius * 0.7, params.radius, dist);
  let sx = x + params.ox; let sy = y + params.oy;
  let src = sample_bilinear_f32(sx, sy);
  let bg = input[idx];
  let a = falloff * params.opacity;
  output[idx] = vec4<f32>(mix(bg.rgb, src.rgb, vec3<f32>(a)), bg.w);
}
"#;

impl Filter for CloneStamp {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        let ox = self.offset_x * width as f32;
        let oy = self.offset_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx*dx + dy*dy).sqrt();
                if dist >= self.radius { continue; }
                let falloff = 1.0 - smoothstep_f32(self.radius * 0.7, self.radius, dist);
                let src = sample_bilinear(input, width, height, x as f32 + ox, y as f32 + oy);
                let i = ((y * width + x) * 4) as usize;
                let a = falloff * self.opacity;
                out[i] = out[i] * (1.0 - a) + src[0] * a;
                out[i+1] = out[i+1] * (1.0 - a) + src[1] * a;
                out[i+2] = out[i+2] * (1.0 - a) + src[2] * a;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{CLONE_STAMP_WGSL}");
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x * _w as f32);
        gpu_push_f32(&mut p, self.center_y * _h as f32);
        gpu_push_f32(&mut p, self.offset_x * _w as f32);
        gpu_push_f32(&mut p, self.offset_y * _h as f32);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.opacity);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}
