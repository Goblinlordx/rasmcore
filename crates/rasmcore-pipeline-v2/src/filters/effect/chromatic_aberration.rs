use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{clamp_coord, gpu_params_push_f32, gpu_params_push_u32};

/// Chromatic aberration — radial R/B channel displacement.
///
/// R channel shifts away from center, B channel shifts toward center.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "chromatic_aberration", category = "effect", cost = "O(n)")]
pub struct ChromaticAberration {
    #[param(min = 0.0, max = 50.0, default = 5.0)]
    pub strength: f32,
}

impl Filter for ChromaticAberration {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt().max(1.0);
        let strength = self.strength / max_dist;

        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let shift = dist * strength;

                // Red: shift outward
                let rx = clamp_coord((x as f32 + dx * shift / dist.max(1.0)).round() as i32, w);
                let ry = clamp_coord((y as f32 + dy * shift / dist.max(1.0)).round() as i32, h);
                out[idx] = input[(ry * w + rx) * 4];

                // Green: no shift
                out[idx + 1] = input[idx + 1];

                // Blue: shift inward
                let bx = clamp_coord((x as f32 - dx * shift / dist.max(1.0)).round() as i32, w);
                let by = clamp_coord((y as f32 - dy * shift / dist.max(1.0)).round() as i32, h);
                out[idx + 2] = input[(by * w + bx) * 4 + 2];

                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }
}

pub(crate) const CHROMATIC_ABERRATION_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  strength: f32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let cx = f32(params.width) / 2.0;
  let cy = f32(params.height) / 2.0;
  let dx = f32(gid.x) - cx;
  let dy = f32(gid.y) - cy;
  let max_d = sqrt(cx * cx + cy * cy);
  let dist = sqrt(dx * dx + dy * dy);
  let shift = params.strength * dist / max(max_d, 1.0);
  let dir_x = select(dx / dist, 0.0, dist < 0.001);
  let dir_y = select(dy / dist, 0.0, dist < 0.001);
  let w = i32(params.width);
  let h = i32(params.height);
  // Red: shift outward
  let rrx = clamp(i32(f32(gid.x) + dir_x * shift), 0, w - 1);
  let rry = clamp(i32(f32(gid.y) + dir_y * shift), 0, h - 1);
  // Blue: shift inward
  let brx = clamp(i32(f32(gid.x) - dir_x * shift), 0, w - 1);
  let bry = clamp(i32(f32(gid.y) - dir_y * shift), 0, h - 1);
  let rp = load_pixel(u32(rry) * params.width + u32(rrx));
  let pixel = load_pixel(idx);
  let bp = load_pixel(u32(bry) * params.width + u32(brx));
  store_pixel(idx, vec4<f32>(rp.x, pixel.y, bp.z, pixel.w));
}
"#;

impl GpuFilter for ChromaticAberration {
    fn shader_body(&self) -> &str {
        CHROMATIC_ABERRATION_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.strength);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}
