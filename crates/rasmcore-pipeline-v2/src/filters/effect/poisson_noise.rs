use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;
use noise::{Rng, SEED_POISSON_NOISE};

use super::{gpu_params_push_f32, gpu_params_push_u32};

/// Poisson noise — signal-dependent noise (brighter regions get more).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "poisson_noise", category = "effect", cost = "O(n)")]
pub struct PoissonNoise {
    #[param(min = 0.1, max = 1000.0, default = 100.0)]
    pub scale: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for PoissonNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        if self.scale <= 0.0 {
            return Ok(input.to_vec());
        }
        let mut rng = Rng::with_offset(self.seed, SEED_POISSON_NOISE);
        let mut out = input.to_vec();
        let inv_scale = 1.0 / self.scale;

        for pixel in out.chunks_exact_mut(4) {
            #[allow(clippy::needless_range_loop)]
            for c in 0..3 {
                let lambda = (pixel[c].max(0.0) * self.scale).max(0.001);
                // Poisson approximation via Gaussian for lambda > 10
                let noisy = if lambda > 10.0 {
                    lambda + lambda.sqrt() * rng.next_gaussian()
                } else {
                    // Knuth algorithm for small lambda
                    let l = (-lambda).exp();
                    let mut k = 0.0f32;
                    let mut p = 1.0f32;
                    loop {
                        k += 1.0;
                        p *= rng.next_f32();
                        if p <= l {
                            break;
                        }
                    }
                    k - 1.0
                };
                pixel[c] = noisy * inv_scale;
            }
        }
        Ok(out)
    }
}

pub(crate) const POISSON_NOISE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  scale: f32,
  inv_scale: f32,
  seed_lo: u32,
  seed_hi: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);
  // Gaussian approx for Poisson: noisy = lambda + sqrt(lambda) * gaussian
  var result = pixel;
  for (var c = 0u; c < 3u; c = c + 1u) {
    let val = select(pixel.x, select(pixel.y, pixel.z, c == 2u), c >= 1u);
    let lambda = max(val, 0.0) * params.scale;
    let u1 = max(abs(noise_2d(idx * 3u + c, 0u, params.seed_lo, params.seed_hi) * 0.5 + 0.5), 0.00001);
    let u2 = noise_2d(idx * 3u + c, 1u, params.seed_lo + 5u, params.seed_hi) * 0.5 + 0.5;
    let g = sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
    let noisy = max(lambda + sqrt(max(lambda, 0.001)) * g, 0.0) * params.inv_scale;
    switch c {
      case 0u: { result.x = noisy; }
      case 1u: { result.y = noisy; }
      case 2u: { result.z = noisy; }
      default: {}
    }
  }
  store_pixel(idx, result);
}
"#;

impl GpuFilter for PoissonNoise {
    fn shader_body(&self) -> &str {
        POISSON_NOISE_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [256, 1, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed ^ noise::SEED_POISSON_NOISE;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.scale);
        gpu_params_push_f32(&mut buf, 1.0 / self.scale.max(0.001));
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, POISSON_NOISE_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
            setup: None,
        }
    }
}
