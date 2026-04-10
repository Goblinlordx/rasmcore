//! PatternFill generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Pattern Fill (tile a repeating pattern)
// ═══════════════════════════════════════════════════════════════════════════

/// Tile the input image as a repeating pattern. Useful for creating seamless backgrounds.
/// tile_w/tile_h control the pattern repeat period in pixels.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "pattern_fill", category = "generator")]
pub struct PatternFill {
    #[param(min = 4.0, max = 512.0, step = 1.0, default = 64.0)]
    pub tile_w: f32,
    #[param(min = 4.0, max = 512.0, step = 1.0, default = 64.0)]
    pub tile_h: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub offset_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub offset_y: f32,
}

const PATTERN_FILL_WGSL: &str = r#"
struct Params { width: u32, height: u32, tile_w: f32, tile_h: f32, offset_x: f32, offset_y: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height;
  if (idx >= total) { return; }
  let x = f32(idx % params.width);
  let y = f32(idx / params.width);
  let ox = params.offset_x * params.tile_w;
  let oy = params.offset_y * params.tile_h;
  let src_x = u32(((x + ox) % params.tile_w + params.tile_w) % params.tile_w) % params.width;
  let src_y = u32(((y + oy) % params.tile_h + params.tile_h) % params.tile_h) % params.height;
  let src_idx = src_y * params.width + src_x;
  output[idx] = input[src_idx];
}
"#;

impl Filter for PatternFill {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let tw = self.tile_w;
        let th = self.tile_h;
        let ox = self.offset_x * tw;
        let oy = self.offset_y * th;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let sx = ((x as f32 + ox) % tw + tw) % tw;
                let sy = ((y as f32 + oy) % th + th) % th;
                let src_x = (sx as u32).min(width - 1);
                let src_y = (sy as u32).min(height - 1);
                let src_i = ((src_y * width + src_x) * 4) as usize;
                let dst_i = ((y * width + x) * 4) as usize;
                out[dst_i..dst_i + 4].copy_from_slice(&input[src_i..src_i + 4]);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, w: u32, h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(w, h);
        gpu_push_f32(&mut p, self.tile_w);
        gpu_push_f32(&mut p, self.tile_h);
        gpu_push_f32(&mut p, self.offset_x);
        gpu_push_f32(&mut p, self.offset_y);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            PATTERN_FILL_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
