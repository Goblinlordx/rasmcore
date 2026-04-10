use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{clamp_coord, gpu_params_push_f32};

/// Chromatic split — offset RGB channels independently.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "chromatic_split", category = "effect", cost = "O(n)")]
pub struct ChromaticSplit {
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub red_dx: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub red_dy: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub green_dx: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub green_dy: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub blue_dx: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub blue_dy: f32,
}

impl Filter for ChromaticSplit {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;

                // Red channel from offset position
                let rx = clamp_coord((x as f32 + self.red_dx).round() as i32, w);
                let ry = clamp_coord((y as f32 + self.red_dy).round() as i32, h);
                out[idx] = input[(ry * w + rx) * 4];

                // Green channel
                let gx = clamp_coord((x as f32 + self.green_dx).round() as i32, w);
                let gy = clamp_coord((y as f32 + self.green_dy).round() as i32, h);
                out[idx + 1] = input[(gy * w + gx) * 4 + 1];

                // Blue channel
                let bx = clamp_coord((x as f32 + self.blue_dx).round() as i32, w);
                let by = clamp_coord((y as f32 + self.blue_dy).round() as i32, h);
                out[idx + 2] = input[(by * w + bx) * 4 + 2];

                out[idx + 3] = input[idx + 3]; // alpha
            }
        }
        Ok(out)
    }
}

pub(crate) const CHROMATIC_SPLIT_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  red_dx: f32,
  red_dy: f32,
  green_dx: f32,
  green_dy: f32,
  blue_dx: f32,
  blue_dy: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let w = i32(params.width);
  let h = i32(params.height);
  let rx = clamp(i32(gid.x) + i32(params.red_dx), 0, w - 1);
  let ry = clamp(i32(gid.y) + i32(params.red_dy), 0, h - 1);
  let gx = clamp(i32(gid.x) + i32(params.green_dx), 0, w - 1);
  let gy = clamp(i32(gid.y) + i32(params.green_dy), 0, h - 1);
  let bx = clamp(i32(gid.x) + i32(params.blue_dx), 0, w - 1);
  let by = clamp(i32(gid.y) + i32(params.blue_dy), 0, h - 1);
  let rp = load_pixel(u32(ry) * params.width + u32(rx));
  let gp = load_pixel(u32(gy) * params.width + u32(gx));
  let bp = load_pixel(u32(by) * params.width + u32(bx));
  let pixel = load_pixel(idx);
  store_pixel(idx, vec4<f32>(rp.x, gp.y, bp.z, pixel.w));
}
"#;

impl GpuFilter for ChromaticSplit {
    fn shader_body(&self) -> &str {
        CHROMATIC_SPLIT_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.red_dx);
        gpu_params_push_f32(&mut buf, self.red_dy);
        gpu_params_push_f32(&mut buf, self.green_dx);
        gpu_params_push_f32(&mut buf, self.green_dy);
        gpu_params_push_f32(&mut buf, self.blue_dx);
        gpu_params_push_f32(&mut buf, self.blue_dy);
        buf
    }
}
