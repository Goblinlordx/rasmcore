use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;

use super::{clamp_coord, gpu_params_push_f32, gpu_params_push_u32};

/// Mirror kaleidoscope — reflect/mirror segments around axis.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "mirror_kaleidoscope", category = "effect", cost = "O(n)")]
pub struct MirrorKaleidoscope {
    #[param(min = 2, max = 32, default = 4)]
    pub segments: u32,
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle: f32,
    #[param(min = 0, max = 2, default = 0)]
    pub mode: u32, // 0=horizontal, 1=vertical, 2=angular
}

impl Filter for MirrorKaleidoscope {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let segments = self.segments.max(2) as usize;
        let mut out = vec![0.0f32; w * h * 4];

        match self.mode {
            0 => {
                // Horizontal mirror
                let seg_w = w / segments;
                if seg_w == 0 {
                    return Ok(input.to_vec());
                }
                for y in 0..h {
                    for x in 0..w {
                        let seg = x / seg_w;
                        let local_x = x % seg_w;
                        let src_x = if seg.is_multiple_of(2) { local_x } else { seg_w - 1 - local_x };
                        let src_x = src_x.min(seg_w - 1);
                        let src_idx = (y * w + src_x) * 4;
                        let dst_idx = (y * w + x) * 4;
                        out[dst_idx..dst_idx + 4].copy_from_slice(&input[src_idx..src_idx + 4]);
                    }
                }
            }
            1 => {
                // Vertical mirror
                let seg_h = h / segments;
                if seg_h == 0 {
                    return Ok(input.to_vec());
                }
                for y in 0..h {
                    let seg = y / seg_h;
                    let local_y = y % seg_h;
                    let src_y = if seg.is_multiple_of(2) { local_y } else { seg_h - 1 - local_y };
                    let src_y = src_y.min(seg_h - 1);
                    for x in 0..w {
                        let src_idx = (src_y * w + x) * 4;
                        let dst_idx = (y * w + x) * 4;
                        out[dst_idx..dst_idx + 4].copy_from_slice(&input[src_idx..src_idx + 4]);
                    }
                }
            }
            _ => {
                // Angular kaleidoscope
                let cx = w as f32 / 2.0;
                let cy = h as f32 / 2.0;
                let segment_angle = std::f32::consts::TAU / segments as f32;
                let base_angle = self.angle.to_radians();

                for y in 0..h {
                    for x in 0..w {
                        let dx = x as f32 - cx;
                        let dy = y as f32 - cy;
                        let mut angle = dy.atan2(dx) - base_angle;
                        if angle < 0.0 {
                            angle += std::f32::consts::TAU;
                        }

                        let seg = (angle / segment_angle) as usize;
                        let local_angle = angle - seg as f32 * segment_angle;
                        let mapped = if seg.is_multiple_of(2) {
                            local_angle + base_angle
                        } else {
                            (segment_angle - local_angle) + base_angle
                        };

                        let dist = (dx * dx + dy * dy).sqrt();
                        let src_x = (cx + dist * mapped.cos()).round() as i32;
                        let src_y = (cy + dist * mapped.sin()).round() as i32;

                        let src_x = clamp_coord(src_x, w);
                        let src_y = clamp_coord(src_y, h);

                        let src_idx = (src_y * w + src_x) * 4;
                        let dst_idx = (y * w + x) * 4;
                        out[dst_idx..dst_idx + 4].copy_from_slice(&input[src_idx..src_idx + 4]);
                    }
                }
            }
        }
        Ok(out)
    }
}

pub(crate) const MIRROR_KALEIDOSCOPE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  segments: u32,
  angle: f32,
  mode: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var sx = gid.x;
  var sy = gid.y;
  let w = params.width;
  let h = params.height;
  if (params.mode == 0u) {
    // Horizontal mirror
    if (gid.x >= w / 2u) { sx = w - 1u - gid.x; }
  } else if (params.mode == 1u) {
    // Vertical mirror
    if (gid.y >= h / 2u) { sy = h - 1u - gid.y; }
  } else {
    // Angular kaleidoscope
    let cx = f32(w) / 2.0;
    let cy = f32(h) / 2.0;
    let dx = f32(gid.x) - cx;
    let dy = f32(gid.y) - cy;
    let dist = sqrt(dx * dx + dy * dy);
    var angle = atan2(dy, dx) - params.angle;
    let seg_angle = 6.2831853 / f32(max(params.segments, 2u));
    angle = angle - floor(angle / seg_angle) * seg_angle;
    if (angle > seg_angle / 2.0) { angle = seg_angle - angle; }
    angle = angle + params.angle;
    sx = u32(clamp(cx + dist * cos(angle), 0.0, f32(w - 1u)));
    sy = u32(clamp(cy + dist * sin(angle), 0.0, f32(h - 1u)));
  }
  store_pixel(idx, load_pixel(sy * w + sx));
}
"#;

impl GpuFilter for MirrorKaleidoscope {
    fn shader_body(&self) -> &str { MIRROR_KALEIDOSCOPE_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, self.segments);
        gpu_params_push_f32(&mut buf, self.angle);
        gpu_params_push_u32(&mut buf, self.mode);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}
