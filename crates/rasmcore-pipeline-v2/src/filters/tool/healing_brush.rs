//! HealingBrush tool filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32, sample_bilinear};
use super::smoothstep_f32;

// Healing Brush — blend source region with local statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Healing brush — copy texture from source offset, match local color.
/// Combines clone_stamp source with local mean color matching.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "healing_brush", category = "tool")]
pub struct HealingBrush {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_y: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.1)]
    pub offset_x: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub offset_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0, hint = "rc.pixels")]
    pub radius: f32,
}

const HEALING_BRUSH_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, ox: f32, oy: f32, radius: f32, _pad: u32, }
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
  let falloff = 1.0 - smoothstep(params.radius * 0.5, params.radius, dist);
  // Source pixel
  let sx = clamp(u32(round(x + params.ox)), 0u, params.width - 1u);
  let sy = clamp(u32(round(y + params.oy)), 0u, params.height - 1u);
  let src = input[sx + sy * params.width];
  let dst = input[idx];
  // Simple healing: blend source texture with destination color
  let src_luma = src.r * 0.2126 + src.g * 0.7152 + src.b * 0.0722;
  let dst_luma = dst.r * 0.2126 + dst.g * 0.7152 + dst.b * 0.0722;
  let ratio = dst_luma / max(src_luma, 0.001);
  let healed = src.rgb * vec3<f32>(ratio);
  let blended = mix(dst.rgb, healed, vec3<f32>(falloff));
  output[idx] = vec4<f32>(blended, dst.w);
}
"#;

impl Filter for HealingBrush {
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
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= self.radius {
                    continue;
                }
                let falloff = 1.0 - smoothstep_f32(self.radius * 0.5, self.radius, dist);
                let src = sample_bilinear(input, width, height, x as f32 + ox, y as f32 + oy);
                let i = ((y * width + x) * 4) as usize;
                let src_luma = src[0] * 0.2126 + src[1] * 0.7152 + src[2] * 0.0722;
                let dst_luma = out[i] * 0.2126 + out[i + 1] * 0.7152 + out[i + 2] * 0.0722;
                let ratio = dst_luma / src_luma.max(0.001);
                for c in 0..3 {
                    let healed = src[c] * ratio;
                    out[i + c] = out[i + c] * (1.0 - falloff) + healed * falloff;
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x * _w as f32);
        gpu_push_f32(&mut p, self.center_y * _h as f32);
        gpu_push_f32(&mut p, self.offset_x * _w as f32);
        gpu_push_f32(&mut p, self.offset_y * _h as f32);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            HEALING_BRUSH_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
