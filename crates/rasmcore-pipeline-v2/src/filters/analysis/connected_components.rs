//! ConnectedComponents filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32, hsl_to_rgb};

/// Connected components labeling — assigns unique colors to connected regions.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "connected_components", category = "analysis")]
pub struct ConnectedComponents {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub threshold: f32,
}

impl Filter for ConnectedComponents {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize; let h = height as usize;
        let mut labels = vec![0u32; w * h];
        let mut next_label = 1u32;
        // Simple two-pass labeling
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let luma = input[i] * 0.2126 + input[i+1] * 0.7152 + input[i+2] * 0.0722;
                if luma <= self.threshold { continue; }
                let left = if x > 0 { labels[y * w + x - 1] } else { 0 };
                let above = if y > 0 { labels[(y-1) * w + x] } else { 0 };
                if left > 0 { labels[y * w + x] = left; }
                else if above > 0 { labels[y * w + x] = above; }
                else { labels[y * w + x] = next_label; next_label += 1; }
            }
        }
        // Colorize labels
        let mut out = vec![0.0f32; input.len()];
        for y in 0..h {
            for x in 0..w {
                let label = labels[y * w + x];
                let i = (y * w + x) * 4;
                if label > 0 {
                    let hue = (label as f32 * 137.508) % 360.0; // golden angle spacing
                    let (r, g, b) = hsl_to_rgb(hue, 0.8, 0.5);
                    out[i] = r; out[i+1] = g; out[i+2] = b;
                }
                out[i+3] = input[i+3];
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        use crate::node::ReductionBuffer;
        // GPU connected components via iterative label propagation:
        // Each pixel gets label = index+1 if above threshold, 0 otherwise.
        // Each pass: pixel adopts minimum label from its 4-connected neighbors.
        // Converges when no labels change.
        let n = (_w * _h) as usize;
        let label_size = n * 4; // u32 per pixel
        let change_size = 4usize;
        let label_buf_id = 60u32;
        let change_buf_id = 61u32;

        // Init: each above-threshold pixel gets unique label
        let init_wgsl = r#"
struct Params { width: u32, height: u32, threshold: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> labels: array<u32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
  if (luma > params.threshold) { labels[idx] = idx + 1u; }
  else { labels[idx] = 0u; }
  output[idx] = px;
}
"#;
        let mut init_p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut init_p, self.threshold); gpu_push_u32(&mut init_p, 0);

        // Propagate: adopt minimum non-zero label from neighbors
        let prop_wgsl = r#"
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> labels: array<u32>;
@group(0) @binding(4) var<storage, read_write> change_count: array<atomic<u32>>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let my_label = labels[idx];
  if (my_label == 0u) { output[idx] = input[idx]; return; }
  let x = i32(idx % params.width); let y = i32(idx / params.width);
  let w = i32(params.width); let h = i32(params.height);
  var min_label = my_label;
  if (x > 0) { let l = labels[u32(x-1) + u32(y) * params.width]; if (l > 0u && l < min_label) { min_label = l; } }
  if (x < w-1) { let l = labels[u32(x+1) + u32(y) * params.width]; if (l > 0u && l < min_label) { min_label = l; } }
  if (y > 0) { let l = labels[u32(x) + u32(y-1) * params.width]; if (l > 0u && l < min_label) { min_label = l; } }
  if (y < h-1) { let l = labels[u32(x) + u32(y+1) * params.width]; if (l > 0u && l < min_label) { min_label = l; } }
  if (min_label != my_label) {
    labels[idx] = min_label;
    atomicAdd(&change_count[0], 1u);
  }
  // Colorize by label (golden angle HSL)
  let hue = f32(min_label % 256u) * 1.618 * 360.0 / 256.0;
  let c = 0.8; let x_c = c * (1.0 - abs(((hue / 60.0) % 2.0) - 1.0));
  let m = 0.1;
  var r: f32; var g: f32; var b: f32;
  let sector = u32(hue / 60.0) % 6u;
  switch (sector) {
    case 0u: { r = c; g = x_c; b = 0.0; }
    case 1u: { r = x_c; g = c; b = 0.0; }
    case 2u: { r = 0.0; g = c; b = x_c; }
    case 3u: { r = 0.0; g = x_c; b = c; }
    case 4u: { r = x_c; g = 0.0; b = c; }
    default: { r = c; g = 0.0; b = x_c; }
  }
  output[idx] = vec4<f32>(r + m, g + m, b + m, input[idx].w);
}
"#;
        let mut prop_p = gpu_params_wh(_w, _h);
        gpu_push_u32(&mut prop_p, 0); gpu_push_u32(&mut prop_p, 0);

        let max_iters = (_w.max(_h) / 2).max(50);
        let mut passes = vec![
            GpuShader {
                body: init_wgsl.to_string(), entry_point: "main", workgroup_size: [256, 1, 1],
                params: init_p, extra_buffers: vec![], convergence_check: None,
            loop_dispatch: None,
                reduction_buffers: vec![ReductionBuffer { id: label_buf_id, initial_data: vec![0u8; label_size], read_write: true }],
            },
        ];
        for _ in 0..max_iters {
            passes.push(GpuShader {
                body: prop_wgsl.to_string(), entry_point: "main", workgroup_size: [256, 1, 1],
                params: prop_p.clone(), extra_buffers: vec![], convergence_check: Some(change_buf_id),
                loop_dispatch: None,
                reduction_buffers: vec![
                    ReductionBuffer { id: label_buf_id, initial_data: vec![], read_write: true },
                    ReductionBuffer { id: change_buf_id, initial_data: vec![0u8; change_size], read_write: true },
                ],
            });
        }
        Some(passes)
    }
}
