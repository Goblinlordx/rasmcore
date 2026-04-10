//! CloudNoise generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::{fbm_cpu, worley_fbm_cpu};

// Cloud Noise (Worley + value noise blend)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate cloud-like noise by blending Worley (cellular) noise with value noise.
/// The worley_blend parameter controls the mix: 0 = pure value, 1 = pure Worley.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "cloud_noise", category = "generator")]
pub struct CloudNoise {
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 60.0)]
    pub scale: f32,
    #[param(min = 1, max = 8, step = 1, default = 5)]
    pub octaves: u32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub persistence: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.4)]
    pub worley_blend: f32,
    #[param(min = 0, max = 99999, step = 1, default = 42, hint = "rc.seed")]
    pub seed: u32,
}

const CLOUD_NOISE_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, octaves: u32, persistence: f32, worley_blend: f32, seed: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn hash2(p: vec2<f32>) -> f32 {
  let k = vec2<f32>(0.3183099, 0.3678794);
  let pp = p * k + k.yx;
  return fract(16.0 * k.x * fract(pp.x * pp.y * (pp.x + pp.y)));
}

fn hash2v(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(hash2(p), hash2(p + vec2<f32>(127.1, 311.7)));
}

fn noise2(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
    mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
    u.y
  );
}

fn worley(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  var min_dist = 1.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let neighbor = vec2<f32>(f32(dx), f32(dy));
      let point = hash2v(i + neighbor);
      let diff = neighbor + point - f;
      min_dist = min(min_dist, length(diff));
    }
  }
  return min_dist;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let base = vec2<f32>(x, y) / params.scale + vec2<f32>(f32(params.seed) * 0.1, f32(params.seed) * 0.07);

  var val = 0.0; var wor = 0.0;
  var amp = 1.0; var freq = 1.0; var total = 0.0;
  for (var i = 0u; i < params.octaves; i++) {
    val += noise2(base * freq) * amp;
    wor += (1.0 - worley(base * freq)) * amp;
    total += amp;
    amp *= params.persistence;
    freq *= 2.0;
  }
  val /= total; wor /= total;
  let v = mix(val, wor, params.worley_blend);
  output[idx] = vec4<f32>(v, v, v, 1.0);
}
"#;

impl Filter for CloudNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let sx = self.seed as f32 * 0.1;
        let sy = self.seed as f32 * 0.07;
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / self.scale + sx;
                let py = y as f32 / self.scale + sy;
                let val = fbm_cpu(px, py, self.octaves, self.persistence);
                let wor = worley_fbm_cpu(px, py, self.octaves, self.persistence);
                let v = val * (1.0 - self.worley_blend) + wor * self.worley_blend;
                let i = ((y * width + x) * 4) as usize;
                out[i] = v;
                out[i + 1] = v;
                out[i + 2] = v;
                out[i + 3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, w: u32, h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(w, h);
        gpu_push_f32(&mut p, self.scale);
        gpu_push_u32(&mut p, self.octaves);
        gpu_push_f32(&mut p, self.persistence);
        gpu_push_f32(&mut p, self.worley_blend);
        gpu_push_u32(&mut p, self.seed);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            CLOUD_NOISE_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
