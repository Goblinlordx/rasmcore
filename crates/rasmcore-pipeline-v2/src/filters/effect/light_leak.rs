use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{gpu_params_push_f32, gpu_params_push_u32};

/// Light leak — procedural warm-toned radial gradient with screen blend.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "light_leak", category = "effect", cost = "O(n)")]
pub struct LightLeak {
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub intensity: f32,
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub position_x: f32,
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub position_y: f32,
    #[param(min = 0.0, max = 2.0, default = 0.5)]
    pub radius: f32,
    #[param(min = 0.0, max = 1.0, default = 0.8)]
    pub warmth: f32,
}

impl Filter for LightLeak {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let cx = self.position_x * w as f32;
        let cy = self.position_y * h as f32;
        let max_r = self.radius * (w.max(h) as f32);
        let inv_r = if max_r > 0.0 { 1.0 / max_r } else { 0.0 };

        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt() * inv_r;
                let falloff = (1.0 - dist).max(0.0);
                let falloff = falloff * falloff; // quadratic

                // Warm-toned leak color (orangeish)
                let leak_r = 1.0 * self.warmth + 0.8 * (1.0 - self.warmth);
                let leak_g = 0.6 * self.warmth + 0.4 * (1.0 - self.warmth);
                let leak_b = 0.2 * self.warmth + 0.1 * (1.0 - self.warmth);

                let strength = falloff * self.intensity;
                let idx = (y * w + x) * 4;

                // Screen blend: a + b - a*b
                out[idx] = out[idx] + leak_r * strength - out[idx] * leak_r * strength;
                out[idx + 1] = out[idx + 1] + leak_g * strength - out[idx + 1] * leak_g * strength;
                out[idx + 2] = out[idx + 2] + leak_b * strength - out[idx + 2] * leak_b * strength;
            }
        }
        Ok(out)
    }
}

pub(crate) const LIGHT_LEAK_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  intensity: f32,
  pos_x: f32,
  pos_y: f32,
  radius: f32,
  warmth: f32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let nx = f32(gid.x) / f32(params.width);
  let ny = f32(gid.y) / f32(params.height);
  let dx = nx - params.pos_x;
  let dy = ny - params.pos_y;
  let dist = sqrt(dx * dx + dy * dy);
  let falloff = max(1.0 - dist / max(params.radius, 0.001), 0.0);
  let leak = falloff * falloff * params.intensity;
  // Screen blend: 1 - (1-a)*(1-b)
  let lr = 1.0 * params.warmth;
  let lg = 0.8 * params.warmth;
  let lb = 0.3 * params.warmth;
  let r = 1.0 - (1.0 - pixel.x) * (1.0 - lr * leak);
  let g = 1.0 - (1.0 - pixel.y) * (1.0 - lg * leak);
  let b = 1.0 - (1.0 - pixel.z) * (1.0 - lb * leak);
  store_pixel(idx, vec4<f32>(r, g, b, pixel.w));
}
"#;

impl GpuFilter for LightLeak {
    fn shader_body(&self) -> &str { LIGHT_LEAK_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.intensity);
        gpu_params_push_f32(&mut buf, self.position_x);
        gpu_params_push_f32(&mut buf, self.position_y);
        gpu_params_push_f32(&mut buf, self.radius);
        gpu_params_push_f32(&mut buf, self.warmth);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}
