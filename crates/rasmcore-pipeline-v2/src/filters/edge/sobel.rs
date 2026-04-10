//! Sobel filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::convolve3x3;

/// Sobel edge detection — 3x3 gradient magnitude.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sobel", category = "spatial", cost = "O(n)")]
pub struct Sobel {
    /// Output scale (1.0 = standard, higher = more visible edges)
    #[param(min = 0.1, max = 5.0, step = 0.1, default = 1.0)]
    pub scale: f32,
}

pub(crate) const SOBEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
pub(crate) const SOBEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

impl Filter for Sobel {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (width as usize, height as usize);
        let mut out = vec![0.0f32; w * h * 4];
        let scale = self.scale;
        for y in 0..h {
            for x in 0..w {
                let gx = convolve3x3(input, w, h, x as i32, y as i32, &SOBEL_X);
                let gy = convolve3x3(input, w, h, x as i32, y as i32, &SOBEL_Y);
                let v = (gx * gx + gy * gy).sqrt() * scale;
                let idx = (y * w + x) * 4;
                out[idx] = v;
                out[idx + 1] = v;
                out[idx + 2] = v;
                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 {
        1
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(SOBEL_WGSL)
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(12);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.scale.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // pad to 16
        Some(buf)
    }
}

const SOBEL_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, _pad: u32, }
@group(0) @binding(2) var<uniform> params: Params;

fn sample_luma(x: i32, y: i32) -> f32 {
    let sx = clamp(x, 0, i32(params.width) - 1);
    let sy = clamp(y, 0, i32(params.height) - 1);
    let p = load_pixel(u32(sy) * params.width + u32(sx));
    return 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let gx = -sample_luma(x-1,y-1) + sample_luma(x+1,y-1)
           - 2.0*sample_luma(x-1,y) + 2.0*sample_luma(x+1,y)
           - sample_luma(x-1,y+1) + sample_luma(x+1,y+1);
    let gy = -sample_luma(x-1,y-1) - 2.0*sample_luma(x,y-1) - sample_luma(x+1,y-1)
           + sample_luma(x-1,y+1) + 2.0*sample_luma(x,y+1) + sample_luma(x+1,y+1);
    let mag = sqrt(gx*gx + gy*gy) * params.scale;
    let idx = gid.y * params.width + gid.x;
    let a = load_pixel(idx).w;
    store_pixel(idx, vec4(mag, mag, mag, a));
}
"#;
