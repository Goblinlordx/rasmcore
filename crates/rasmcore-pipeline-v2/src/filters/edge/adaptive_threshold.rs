//! AdaptiveThreshold filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::luminance;

/// Adaptive threshold — per-pixel threshold based on local mean.
/// Each pixel is compared to the mean luminance in a surrounding window.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "adaptive_threshold", category = "spatial", cost = "O(n * r^2)")]
pub struct AdaptiveThreshold {
    /// Window radius for local mean computation
    #[param(min = 1, max = 50, step = 1, default = 5)]
    pub radius: u32,
    /// Offset subtracted from local mean (negative = more white, positive = more black)
    #[param(min = -0.5, max = 0.5, step = 0.02, default = 0.02)]
    pub offset: f32,
}

impl Filter for AdaptiveThreshold {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (width as usize, height as usize);
        let r = self.radius as i32;
        let offset = self.offset;

        // Compute integral image of luminance for O(1) local mean
        let mut integral = vec![0.0f64; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row_sum = 0.0f64;
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let l = luminance(input[idx], input[idx + 1], input[idx + 2]) as f64;
                row_sum += l;
                integral[(y + 1) * (w + 1) + (x + 1)] = row_sum + integral[y * (w + 1) + (x + 1)];
            }
        }

        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let x1 = (x - r).max(0) as usize;
                let y1 = (y - r).max(0) as usize;
                let x2 = (x + r + 1).min(w as i32) as usize;
                let y2 = (y + r + 1).min(h as i32) as usize;
                let area = ((x2 - x1) * (y2 - y1)) as f64;
                let sum = integral[y2 * (w + 1) + x2] - integral[y1 * (w + 1) + x2]
                        - integral[y2 * (w + 1) + x1] + integral[y1 * (w + 1) + x1];
                let local_mean = (sum / area) as f32;

                let idx = (y as usize * w + x as usize) * 4;
                let l = luminance(input[idx], input[idx + 1], input[idx + 2]);
                let v = if l >= local_mean - offset { 1.0 } else { 0.0 };
                out[idx] = v; out[idx + 1] = v; out[idx + 2] = v;
                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 { self.radius }

    fn gpu_shader_body(&self) -> Option<&'static str> { Some(ADAPTIVE_THRESHOLD_WGSL) }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.radius.to_le_bytes());
        buf.extend_from_slice(&self.offset.to_le_bytes());
        Some(buf)
    }
}

/// Adaptive threshold GPU — brute-force local mean within radius window.
/// GPU parallelism makes the O(r^2) per-pixel cost fast for typical radii (5-50).
/// For very large radii a prefix-sum approach would be better, but this covers
/// the practical range.
const ADAPTIVE_THRESHOLD_WGSL: &str = r#"
struct Params { width: u32, height: u32, radius: u32, offset: f32, }
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

    let r = i32(params.radius);
    var sum = 0.0;
    var count = 0.0;
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            sum += sample_luma(x + dx, y + dy);
            count += 1.0;
        }
    }
    let local_mean = sum / count;
    let l = sample_luma(x, y);
    let v = select(0.0, 1.0, l >= local_mean - params.offset);
    let idx = gid.y * params.width + gid.x;
    store_pixel(idx, vec4(v, v, v, load_pixel(idx).w));
}
"#;
