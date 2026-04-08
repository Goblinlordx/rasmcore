//! HarrisCorners filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Harris Corners — corner strength map
// ═══════════════════════════════════════════════════════════════════════════

/// Harris corner detection — outputs corner strength as grayscale.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "harris_corners", category = "analysis")]
pub struct HarrisCorners {
    #[param(min = 0.01, max = 0.3, step = 0.01, default = 0.04)]
    pub k: f32,
    #[param(min = 1, max = 5, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

const HARRIS_WGSL: &str = r#"
struct Params { width: u32, height: u32, k: f32, radius: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let r = i32(params.radius);
  var ixx: f32 = 0.0; var iyy: f32 = 0.0; var ixy: f32 = 0.0;
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let sx = clamp(x + dx, 0, w - 1); let sy = clamp(y + dy, 0, h - 1);
      let sxp = clamp(x + dx + 1, 0, w - 1); let syp = clamp(y + dy + 1, 0, h - 1);
      let sxm = clamp(x + dx - 1, 0, w - 1); let sym = clamp(y + dy - 1, 0, h - 1);
      let c = input[u32(sx) + u32(sy) * params.width];
      let cx1 = input[u32(sxp) + u32(sy) * params.width];
      let cx0 = input[u32(sxm) + u32(sy) * params.width];
      let cy1 = input[u32(sx) + u32(syp) * params.width];
      let cy0 = input[u32(sx) + u32(sym) * params.width];
      let luma = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
      let gx = (cx1.r * 0.2126 + cx1.g * 0.7152 + cx1.b * 0.0722) - (cx0.r * 0.2126 + cx0.g * 0.7152 + cx0.b * 0.0722);
      let gy = (cy1.r * 0.2126 + cy1.g * 0.7152 + cy1.b * 0.0722) - (cy0.r * 0.2126 + cy0.g * 0.7152 + cy0.b * 0.0722);
      ixx += gx * gx; iyy += gy * gy; ixy += gx * gy;
    }
  }
  let det = ixx * iyy - ixy * ixy;
  let trace = ixx + iyy;
  let response = det - params.k * trace * trace;
  let v = clamp(response * 1000.0, 0.0, 1.0);
  let orig = input[u32(x) + u32(y) * params.width];
  output[u32(x) + u32(y) * params.width] = vec4<f32>(v, v, v, orig.w);
}
"#;

impl Filter for HarrisCorners {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let w = width as i32; let h = height as i32; let r = self.radius as i32;
        for y in 0..h {
            for x in 0..w {
                let mut ixx = 0.0f32; let mut iyy = 0.0f32; let mut ixy = 0.0f32;
                for dy in -r..=r { for dx in -r..=r {
                    let sx = (x+dx).max(0).min(w-1) as usize;
                    let sy = (y+dy).max(0).min(h-1) as usize;
                    let sxp = (x+dx+1).max(0).min(w-1) as usize;
                    let sxm = (x+dx-1).max(0).min(w-1) as usize;
                    let syp = (y+dy+1).max(0).min(h-1) as usize;
                    let sym = (y+dy-1).max(0).min(h-1) as usize;
                    let wi = width as usize;
                    let lx1 = input[(sy*wi+sxp)*4]*0.2126 + input[(sy*wi+sxp)*4+1]*0.7152 + input[(sy*wi+sxp)*4+2]*0.0722;
                    let lx0 = input[(sy*wi+sxm)*4]*0.2126 + input[(sy*wi+sxm)*4+1]*0.7152 + input[(sy*wi+sxm)*4+2]*0.0722;
                    let ly1 = input[(syp*wi+sx)*4]*0.2126 + input[(syp*wi+sx)*4+1]*0.7152 + input[(syp*wi+sx)*4+2]*0.0722;
                    let ly0 = input[(sym*wi+sx)*4]*0.2126 + input[(sym*wi+sx)*4+1]*0.7152 + input[(sym*wi+sx)*4+2]*0.0722;
                    let gx = lx1 - lx0; let gy = ly1 - ly0;
                    ixx += gx*gx; iyy += gy*gy; ixy += gx*gy;
                }}
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let response = det - self.k * trace * trace;
                let v = (response * 1000.0).max(0.0).min(1.0);
                let i = ((y * w + x) * 4) as usize;
                out[i] = v; out[i+1] = v; out[i+2] = v;
                out[i+3] = input[i+3];
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.k); gpu_push_u32(&mut p, self.radius);
        Some(vec![GpuShader::new(HARRIS_WGSL.to_string(), "main", [16, 16, 1], p)])
    }

    fn tile_overlap(&self) -> u32 { self.radius + 1 }
}
