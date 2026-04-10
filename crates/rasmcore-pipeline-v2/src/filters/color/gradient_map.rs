use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::build_gradient_lut;
use crate::filters::helpers::luminance;

/// Gradient map — map luminance to a color gradient.
#[derive(Clone)]
pub struct GradientMap {
    /// Gradient stops: (position [0,1], r, g, b) sorted by position.
    pub stops: Vec<(f32, f32, f32, f32)>,
}

impl Filter for GradientMap {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.stops.is_empty() {
            return Ok(input.to_vec());
        }
        // Build 256-entry LUT for fast lookup
        let lut = build_gradient_lut(&self.stops);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let luma = luminance(pixel[0], pixel[1], pixel[2]);
            let idx = (luma * 255.0).round().clamp(0.0, 255.0) as usize;
            pixel[0] = lut[idx * 3];
            pixel[1] = lut[idx * 3 + 1];
            pixel[2] = lut[idx * 3 + 2];
        }
        Ok(out)
    }
}

/// GradientMap GPU — luminance to gradient color via extra buffer LUT (256*3 f32).
const GRADIENT_MAP_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> lut: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);
  let luma = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
  let li = u32(clamp(round(luma * 255.0), 0.0, 255.0));
  store_pixel(idx, vec4<f32>(lut[li * 3u], lut[li * 3u + 1u], lut[li * 3u + 2u], pixel.w));
}
"#;

impl GpuFilter for GradientMap {
    fn shader_body(&self) -> &str {
        GRADIENT_MAP_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [256, 1, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        let lut = build_gradient_lut(&self.stops);
        let mut buf = Vec::with_capacity(lut.len() * 4);
        for v in &lut {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        vec![buf]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::helpers::luminance;

    fn assert_rgb_close(actual: &[f32], expected: (f32, f32, f32), tol: f32, label: &str) {
        assert!(
            (actual[0] - expected.0).abs() < tol
                && (actual[1] - expected.1).abs() < tol
                && (actual[2] - expected.2).abs() < tol,
            "{label}: expected ({:.4}, {:.4}, {:.4}), got ({:.4}, {:.4}, {:.4})",
            expected.0,
            expected.1,
            expected.2,
            actual[0],
            actual[1],
            actual[2]
        );
    }

    #[test]
    fn gradient_map_bw() {
        // Black → white gradient: should map luma to position
        let stops = vec![(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)];
        let f = GradientMap { stops };
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let out = f.compute(&input, 1, 1).unwrap();
        let luma = luminance(0.5, 0.5, 0.5);
        assert_rgb_close(&out, (luma, luma, luma), 0.02, "gradient bw");
    }
}
