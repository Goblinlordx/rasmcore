use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;
use noise::{Rng, SEED_UNIFORM_NOISE};

use super::{gpu_params_push_f32, gpu_params_push_u32};

/// Uniform noise — additive uniformly-distributed noise.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "uniform_noise", category = "effect", cost = "O(n)")]
pub struct UniformNoise {
    #[param(min = 0.0, max = 255.0, default = 25.0)]
    pub range: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for UniformNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        if self.range <= 0.0 {
            return Ok(input.to_vec());
        }
        let range = self.range / 255.0;
        let mut rng = Rng::with_offset(self.seed, SEED_UNIFORM_NOISE);
        let mut out = input.to_vec();

        for pixel in out.chunks_exact_mut(4) {
            pixel[0] += rng.next_f32_signed() * range;
            pixel[1] += rng.next_f32_signed() * range;
            pixel[2] += rng.next_f32_signed() * range;
        }
        Ok(out)
    }
}

pub(crate) const UNIFORM_NOISE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  range: f32,
  seed_lo: u32,
  seed_hi: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);
  let n = noise_2d(idx, 0u, params.seed_lo, params.seed_hi) * params.range;
  store_pixel(idx, vec4<f32>(pixel.x + n, pixel.y + n, pixel.z + n, pixel.w));
}
"#;

impl GpuFilter for UniformNoise {
    fn shader_body(&self) -> &str { UNIFORM_NOISE_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed ^ noise::SEED_UNIFORM_NOISE;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.range / 255.0);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, UNIFORM_NOISE_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}
