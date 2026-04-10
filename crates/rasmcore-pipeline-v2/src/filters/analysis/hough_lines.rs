//! HoughLines filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

/// Hough line detection — output line visualization.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "hough_lines", category = "analysis")]
pub struct HoughLines {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub threshold: f32,
}

const HOUGH_LINES_WGSL: &str = r#"
struct Params { width: u32, height: u32, threshold: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let idx = u32(x) + u32(y) * params.width;
  if (x == 0 || x >= w-1 || y == 0 || y >= h-1) { output[idx] = vec4<f32>(0.0, 0.0, 0.0, input[idx].w); return; }
  let lx1 = input[u32(x+1) + u32(y) * params.width]; let lx0 = input[u32(x-1) + u32(y) * params.width];
  let ly1 = input[u32(x) + u32(y+1) * params.width]; let ly0 = input[u32(x) + u32(y-1) * params.width];
  let gx = (lx1.r * 0.2126 + lx1.g * 0.7152 + lx1.b * 0.0722) - (lx0.r * 0.2126 + lx0.g * 0.7152 + lx0.b * 0.0722);
  let gy = (ly1.r * 0.2126 + ly1.g * 0.7152 + ly1.b * 0.0722) - (ly0.r * 0.2126 + ly0.g * 0.7152 + ly0.b * 0.0722);
  let edge = sqrt(gx*gx + gy*gy);
  let v = select(0.0, 1.0, edge > params.threshold);
  output[idx] = vec4<f32>(v, v, v, input[idx].w);
}
"#;

impl Filter for HoughLines {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let w = width as i32;
        let h = height as i32;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = (y * w + x) as usize * 4;
                let lx = input[((y * w + x + 1) * 4) as usize] * 0.2126
                    + input[((y * w + x + 1) * 4 + 1) as usize] * 0.7152
                    + input[((y * w + x + 1) * 4 + 2) as usize] * 0.0722;
                let lm = input[((y * w + x - 1) * 4) as usize] * 0.2126
                    + input[((y * w + x - 1) * 4 + 1) as usize] * 0.7152
                    + input[((y * w + x - 1) * 4 + 2) as usize] * 0.0722;
                let edge = (lx - lm).abs();
                let v = if edge > self.threshold { 1.0 } else { 0.0 };
                out[i] = v;
                out[i + 1] = v;
                out[i + 2] = v;
                out[i + 3] = input[i + 3];
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.threshold);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            HOUGH_LINES_WGSL.to_string(),
            "main",
            [16, 16, 1],
            p,
        )])
    }

    fn tile_overlap(&self) -> u32 {
        1
    }
}
