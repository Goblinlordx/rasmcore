use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;
use noise::{Rng, SEED_SALT_PEPPER};

use super::{gpu_params_push_f32, gpu_params_push_u32};

/// Salt-and-pepper noise — randomly replace pixels with black or white.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "salt_pepper_noise", category = "effect", cost = "O(n)")]
pub struct SaltPepperNoise {
    #[param(min = 0.0, max = 1.0, default = 0.05)]
    pub density: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for SaltPepperNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut rng = Rng::with_offset(self.seed, SEED_SALT_PEPPER);
        let mut out = input.to_vec();

        for pixel in out.chunks_exact_mut(4) {
            if rng.next_f32() < self.density {
                let val = if rng.next_f32() < 0.5 { 0.0 } else { 1.0 };
                pixel[0] = val;
                pixel[1] = val;
                pixel[2] = val;
            }
        }
        Ok(out)
    }
}

pub(crate) const SALT_PEPPER_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  density: f32,
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
  let r = noise_2d(idx, 0u, params.seed_lo, params.seed_hi) * 0.5 + 0.5;
  if (r < params.density) {
    let bw = noise_2d(idx, 1u, params.seed_lo + 3u, params.seed_hi) * 0.5 + 0.5;
    let val = select(0.0, 1.0, bw > 0.5);
    store_pixel(idx, vec4<f32>(val, val, val, pixel.w));
  } else {
    store_pixel(idx, pixel);
  }
}
"#;

impl GpuFilter for SaltPepperNoise {
    fn shader_body(&self) -> &str { SALT_PEPPER_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed ^ noise::SEED_SALT_PEPPER;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.density);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, SALT_PEPPER_WGSL),
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
