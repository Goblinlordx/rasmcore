//! PerlinNoise generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::fbm_cpu;

// Perlin Noise
// ═══════════════════════════════════════════════════════════════════════════

/// Generate Perlin noise pattern.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "perlin_noise", category = "generator")]
pub struct PerlinNoise {
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 50.0)]
    pub scale: f32,
    #[param(min = 1, max = 8, step = 1, default = 4)]
    pub octaves: u32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub persistence: f32,
    #[param(min = 0, max = 99999, step = 1, default = 42, hint = "rc.seed")]
    pub seed: u32,
}

pub(super) const PERLIN_NOISE_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, octaves: u32, persistence: f32, seed: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn hash2(p: vec2<f32>) -> f32 {
  let k = vec2<f32>(0.3183099, 0.3678794);
  let pp = p * k + k.yx;
  return fract(16.0 * k.x * fract(pp.x * pp.y * (pp.x + pp.y)));
}

fn noise2(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i + vec2<f32>(0.0, 0.0)), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
    mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
    u.y
  );
}

fn fbm(p: vec2<f32>, octaves: u32, persistence: f32) -> f32 {
  var value = 0.0; var amplitude = 1.0; var frequency = 1.0; var total_amp = 0.0;
  for (var i = 0u; i < octaves; i = i + 1u) {
    value += noise2(p * frequency) * amplitude;
    total_amp += amplitude;
    amplitude *= persistence;
    frequency *= 2.0;
  }
  return value / total_amp;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let p = vec2<f32>(x, y) / params.scale + vec2<f32>(f32(params.seed) * 0.1, f32(params.seed) * 0.07);
  let v = fbm(p, params.octaves, params.persistence);
  output[idx] = vec4<f32>(v, v, v, 1.0);
}
"#;

impl Filter for PerlinNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let seed_offset_x = self.seed as f32 * 0.1;
        let seed_offset_y = self.seed as f32 * 0.07;
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / self.scale + seed_offset_x;
                let py = y as f32 / self.scale + seed_offset_y;
                let v = fbm_cpu(px, py, self.octaves, self.persistence);
                let i = ((y * width + x) * 4) as usize;
                out[i] = v;
                out[i + 1] = v;
                out[i + 2] = v;
                out[i + 3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.scale);
        gpu_push_u32(&mut p, self.octaves);
        gpu_push_f32(&mut p, self.persistence);
        gpu_push_u32(&mut p, self.seed);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            PERLIN_NOISE_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
