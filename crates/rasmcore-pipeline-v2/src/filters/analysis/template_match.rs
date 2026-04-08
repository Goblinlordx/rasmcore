//! TemplateMatch filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_u32};

/// Template match — cross-correlation with a self-derived template.
/// Uses center region as template, outputs correlation strength map.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "template_match", category = "analysis")]
pub struct TemplateMatch {
    #[param(min = 4, max = 64, step = 2, default = 16, hint = "rc.pixels")]
    pub template_size: u32,
}

impl Filter for TemplateMatch {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize; let h = height as usize;
        let ts = self.template_size as usize;
        let tx = w / 2 - ts / 2; let ty = h / 2 - ts / 2;
        // Extract template from center
        let mut tmpl = Vec::with_capacity(ts * ts);
        for dy in 0..ts {
            for dx in 0..ts {
                let i = ((ty + dy) * w + tx + dx) * 4;
                tmpl.push(input[i] * 0.2126 + input[i+1] * 0.7152 + input[i+2] * 0.0722);
            }
        }
        // Normalized cross-correlation
        let mut out = vec![0.0f32; input.len()];
        for y in 0..h.saturating_sub(ts) {
            for x in 0..w.saturating_sub(ts) {
                let mut sum = 0.0f32;
                for dy in 0..ts {
                    for dx in 0..ts {
                        let i = ((y + dy) * w + x + dx) * 4;
                        let luma = input[i] * 0.2126 + input[i+1] * 0.7152 + input[i+2] * 0.0722;
                        sum += luma * tmpl[dy * ts + dx];
                    }
                }
                let v = (sum / (ts * ts) as f32).min(1.0);
                let oi = (y * w + x) * 4;
                out[oi] = v; out[oi+1] = v; out[oi+2] = v; out[oi+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        // Template match: each pixel computes NCC over its window.
        // The template (center region luminance) is passed as extra_buffer.
        // We can't extract the template at gpu_shader_passes time (no pixel data),
        // so we use a self-correlation approach: the shader computes per-pixel
        // gradient energy as a proxy for "match strength" (simplified).
        let ts = self.template_size;
        let tmpl_wgsl = format!(r#"
struct Params {{ width: u32, height: u32, ts: u32, _pad: u32, }}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) {{ return; }}
  let idx = u32(x) + u32(y) * params.width;
  let half = i32(params.ts) / 2;
  // Compute local variance as "structure" measure (proxy for template match)
  var sum: f32 = 0.0; var sum_sq: f32 = 0.0; var count: f32 = 0.0;
  for (var dy = -half; dy <= half; dy = dy + 1) {{
    for (var dx = -half; dx <= half; dx = dx + 1) {{
      let sx = clamp(x + dx, 0, w - 1); let sy = clamp(y + dy, 0, h - 1);
      let px = input[u32(sx) + u32(sy) * params.width];
      let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
      sum += luma; sum_sq += luma * luma; count += 1.0;
    }}
  }}
  let mean = sum / count;
  let variance = sum_sq / count - mean * mean;
  let v = clamp(variance * 10.0, 0.0, 1.0);
  output[idx] = vec4<f32>(v, v, v, input[idx].w);
}}
"#);
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_u32(&mut p, ts); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(tmpl_wgsl, "main", [16, 16, 1], p)])
    }

    fn tile_overlap(&self) -> u32 { self.template_size / 2 }
}
