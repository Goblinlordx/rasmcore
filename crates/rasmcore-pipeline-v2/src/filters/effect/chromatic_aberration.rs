use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{gpu_params_push_f32, gpu_params_push_u32};

/// Chromatic aberration — radial R/B channel displacement with bilinear sampling.
///
/// R channel shifts away from center, B channel shifts toward center.
/// Uses sub-pixel bilinear interpolation (matching professional tools like
/// Lightroom, Photoshop, and Nuke).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "chromatic_aberration", category = "effect", cost = "O(n)")]
pub struct ChromaticAberration {
    #[param(min = 0.0, max = 50.0, default = 5.0)]
    pub strength: f32,
}

/// Bilinear sample a single channel from RGBA interleaved data, clamping at edges.
#[inline]
fn sample_channel(input: &[f32], w: usize, h: usize, x: f32, y: f32, channel: usize) -> f32 {
    let x0 = (x.floor() as isize).max(0).min(w as isize - 1) as usize;
    let y0 = (y.floor() as isize).max(0).min(h as isize - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (x - x.floor()).clamp(0.0, 1.0);
    let fy = (y - y.floor()).clamp(0.0, 1.0);
    let v00 = input[(y0 * w + x0) * 4 + channel];
    let v10 = input[(y0 * w + x1) * 4 + channel];
    let v01 = input[(y1 * w + x0) * 4 + channel];
    let v11 = input[(y1 * w + x1) * 4 + channel];
    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy + v11 * fx * fy
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
                let d = dist.max(1.0);
                let nx = dx / d;
                let ny = dy / d;

                // Red: shift outward (bilinear sub-pixel sampling)
                out[idx] = sample_channel(input, w, h, x as f32 + nx * shift, y as f32 + ny * shift, 0);

                // Green: no shift
                out[idx + 1] = input[idx + 1];

                // Blue: shift inward (bilinear sub-pixel sampling)
                out[idx + 2] = sample_channel(input, w, h, x as f32 - nx * shift, y as f32 - ny * shift, 2);

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

// Bilinear sample a single channel (0=R, 1=G, 2=B) at sub-pixel coordinates
fn sample_ch(sx: f32, sy: f32, ch: u32) -> f32 {
  let w = params.width;
  let h = params.height;
  let x0 = u32(clamp(i32(floor(sx)), 0, i32(w) - 1));
  let y0 = u32(clamp(i32(floor(sy)), 0, i32(h) - 1));
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let fx = clamp(sx - floor(sx), 0.0, 1.0);
  let fy = clamp(sy - floor(sy), 0.0, 1.0);
  let v00 = load_pixel(y0 * w + x0)[ch];
  let v10 = load_pixel(y0 * w + x1)[ch];
  let v01 = load_pixel(y1 * w + x0)[ch];
  let v11 = load_pixel(y1 * w + x1)[ch];
  return v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy)
       + v01 * (1.0 - fx) * fy + v11 * fx * fy;
}

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
  let d = max(dist, 1.0);
  let nx = dx / d;
  let ny = dy / d;
  // Red: shift outward (bilinear)
  let r = sample_ch(f32(gid.x) + nx * shift, f32(gid.y) + ny * shift, 0u);
  // Green: no shift
  let pixel = load_pixel(idx);
  // Blue: shift inward (bilinear)
  let b = sample_ch(f32(gid.x) - nx * shift, f32(gid.y) - ny * shift, 2u);
  store_pixel(idx, vec4<f32>(r, pixel.y, b, pixel.w));
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
