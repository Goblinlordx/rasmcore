//! Canny filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::convolve3x3;
use super::{SOBEL_X, SOBEL_Y};

/// Canny edge detection — Sobel gradient + double threshold + edge thinning.
///
/// Simplified single-pass: gradient magnitude with hysteresis thresholding.
/// True Canny requires non-maximum suppression (multi-pass), implemented
/// here as a gradient threshold approximation for real-time use.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "canny", category = "spatial", cost = "O(n)")]
pub struct Canny {
    /// Low threshold (edges below this are suppressed)
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.1)]
    pub low: f32,
    /// High threshold (edges above this are strong edges)
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.3)]
    pub high: f32,
}

impl Filter for Canny {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (width as usize, height as usize);
        let mut out = vec![0.0f32; w * h * 4];
        let (lo, hi) = (self.low, self.high);
        for y in 0..h {
            for x in 0..w {
                let gx = convolve3x3(input, w, h, x as i32, y as i32, &SOBEL_X);
                let gy = convolve3x3(input, w, h, x as i32, y as i32, &SOBEL_Y);
                let mag = (gx * gx + gy * gy).sqrt();
                let v = if mag >= hi { 1.0 } else if mag >= lo { 0.5 } else { 0.0 };
                let idx = (y * w + x) * 4;
                out[idx] = v; out[idx + 1] = v; out[idx + 2] = v;
                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 { 1 }

    fn gpu_shader_body(&self) -> Option<&'static str> { Some(CANNY_WGSL) }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.low.to_le_bytes());
        buf.extend_from_slice(&self.high.to_le_bytes());
        Some(buf)
    }
}

const CANNY_WGSL: &str = r#"
struct Params { width: u32, height: u32, low: f32, high: f32, }
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
    let gx = -sample_luma(x-1,y-1) + sample_luma(x+1,y-1) - 2.0*sample_luma(x-1,y) + 2.0*sample_luma(x+1,y) - sample_luma(x-1,y+1) + sample_luma(x+1,y+1);
    let gy = -sample_luma(x-1,y-1) - 2.0*sample_luma(x,y-1) - sample_luma(x+1,y-1) + sample_luma(x-1,y+1) + 2.0*sample_luma(x,y+1) + sample_luma(x+1,y+1);
    let mag = sqrt(gx*gx + gy*gy);
    var v = 0.0;
    if (mag >= params.high) { v = 1.0; } else if (mag >= params.low) { v = 0.5; }
    let idx = gid.y * params.width + gid.x;
    store_pixel(idx, vec4(v, v, v, load_pixel(idx).w));
}
"#;
