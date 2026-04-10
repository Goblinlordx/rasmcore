use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;
use noise::SEED_GLITCH;

use super::{clamp_coord, gpu_params_push_f32, gpu_params_push_u32};

/// Glitch — horizontal scanline displacement with RGB channel offset.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "glitch", category = "effect", cost = "O(n)")]
pub struct Glitch {
    #[param(min = 0.0, max = 200.0, default = 20.0)]
    pub shift_amount: f32,
    #[param(min = 0.0, max = 100.0, default = 10.0)]
    pub channel_offset: f32,
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub intensity: f32,
    #[param(min = 1, max = 100, default = 8)]
    pub band_height: u32,
    #[param(min = 0, max = 100, default = 42)]
    pub seed: u32,
}

impl Filter for Glitch {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let band_h = self.band_height.max(1) as usize;
        let mut out = input.to_vec();

        for y in 0..h {
            let band = y / band_h;
            let noise = noise::noise_1d(band as u32, self.seed as u64 ^ SEED_GLITCH);

            if noise.abs() > 1.0 - self.intensity {
                let shift = (noise * self.shift_amount) as i32;
                let ch_off = (noise * self.channel_offset) as i32;

                for x in 0..w {
                    let idx = (y * w + x) * 4;

                    // Red channel with shift + offset
                    let rx = clamp_coord(x as i32 + shift + ch_off, w);
                    out[idx] = input[(y * w + rx) * 4];

                    // Green channel with shift
                    let gx = clamp_coord(x as i32 + shift, w);
                    out[idx + 1] = input[(y * w + gx) * 4 + 1];

                    // Blue channel with shift - offset
                    let bx = clamp_coord(x as i32 + shift - ch_off, w);
                    out[idx + 2] = input[(y * w + bx) * 4 + 2];
                }
            }
        }
        Ok(out)
    }
}

pub(crate) const GLITCH_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  shift_amount: f32,
  channel_offset: f32,
  intensity: f32,
  band_height: u32,
  seed_lo: u32,
  seed_hi: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let band = gid.y / max(params.band_height, 1u);
  let n = noise_2d(band, 0u, params.seed_lo, params.seed_hi);
  if (abs(n) <= 1.0 - params.intensity) {
    store_pixel(idx, pixel);
    return;
  }
  let shift = i32(n * params.shift_amount);
  let ch_off = i32(n * params.channel_offset);
  let w = i32(params.width);
  let rx = clamp(i32(gid.x) + shift + ch_off, 0, w - 1);
  let gx = clamp(i32(gid.x) + shift, 0, w - 1);
  let bx = clamp(i32(gid.x) + shift - ch_off, 0, w - 1);
  let rp = load_pixel(gid.y * params.width + u32(rx));
  let gp = load_pixel(gid.y * params.width + u32(gx));
  let bp = load_pixel(gid.y * params.width + u32(bx));
  store_pixel(idx, vec4<f32>(rp.x, gp.y, bp.z, pixel.w));
}
"#;

impl GpuFilter for Glitch {
    fn shader_body(&self) -> &str {
        GLITCH_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed as u64 ^ noise::SEED_GLITCH;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.shift_amount);
        gpu_params_push_f32(&mut buf, self.channel_offset);
        gpu_params_push_f32(&mut buf, self.intensity);
        gpu_params_push_u32(&mut buf, self.band_height);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, GLITCH_WGSL),
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
