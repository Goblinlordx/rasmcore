use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;
use noise::{Rng, SEED_GAUSSIAN_NOISE};

use super::{gpu_params_push_f32, gpu_params_push_u32};

/// Gaussian noise — additive normally-distributed noise.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gaussian_noise", category = "effect", cost = "O(n)")]
pub struct GaussianNoise {
    #[param(min = 0.0, max = 100.0, default = 10.0)]
    pub amount: f32,
    #[param(min = -255.0, max = 255.0, default = 0.0)]
    pub mean: f32,
    #[param(min = 0.0, max = 255.0, default = 25.0)]
    pub sigma: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for GaussianNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        if self.amount <= 0.0 {
            return Ok(input.to_vec());
        }
        let amount = self.amount / 100.0;
        let sigma = self.sigma / 255.0; // normalize to [0,1] range
        let mean = self.mean / 255.0;
        let mut rng = Rng::with_offset(self.seed, SEED_GAUSSIAN_NOISE);
        let mut out = input.to_vec();

        for pixel in out.chunks_exact_mut(4) {
            let n0 = (mean + sigma * rng.next_gaussian()) * amount;
            let n1 = (mean + sigma * rng.next_gaussian()) * amount;
            let n2 = (mean + sigma * rng.next_gaussian()) * amount;
            pixel[0] += n0;
            pixel[1] += n1;
            pixel[2] += n2;
        }
        Ok(out)
    }
}

pub(crate) const GAUSSIAN_NOISE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  amount: f32,
  mean: f32,
  sigma: f32,
  seed_lo: u32,
  seed_hi: u32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);
  // Box-Muller: two uniform -> one gaussian
  let u1 = max(abs(noise_2d(idx, 0u, params.seed_lo, params.seed_hi) * 0.5 + 0.5), 0.00001);
  let u2 = noise_2d(idx, 1u, params.seed_lo + 7u, params.seed_hi) * 0.5 + 0.5;
  let g = sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
  let n = (params.mean + params.sigma * g) * params.amount;
  store_pixel(idx, vec4<f32>(pixel.x + n, pixel.y + n, pixel.z + n, pixel.w));
}
"#;

impl GpuFilter for GaussianNoise {
    fn shader_body(&self) -> &str { GAUSSIAN_NOISE_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed ^ noise::SEED_GAUSSIAN_NOISE;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.amount / 100.0);
        gpu_params_push_f32(&mut buf, self.mean / 255.0);
        gpu_params_push_f32(&mut buf, self.sigma / 255.0);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0); // pad
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, GAUSSIAN_NOISE_WGSL),
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
