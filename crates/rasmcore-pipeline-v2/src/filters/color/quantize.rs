use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::{median_cut_palette, nearest_color};

/// Median-cut color quantization (no dithering).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "quantize", category = "color", cost = "O(n)")]
pub struct Quantize {
    /// Max palette size (2-256).
    #[param(min = 2, max = 256, default = 16)]
    pub max_colors: u32,
}

impl Filter for Quantize {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let palette = median_cut_palette(input, self.max_colors as usize);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = nearest_color(&palette, pixel[0], pixel[1], pixel[2]);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

/// Quantize GPU — nearest-color from precomputed palette.
/// The palette is computed on CPU (median-cut), passed as extra_buffers[0].
/// GPU does the expensive per-pixel nearest-color search in parallel.
const QUANTIZE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  palette_size: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> palette: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = input[idx];
  var best_d = 1e30;
  var best_r = 0.0;
  var best_g = 0.0;
  var best_b = 0.0;
  for (var i = 0u; i < params.palette_size; i = i + 1u) {
    let pr = palette[i * 3u];
    let pg = palette[i * 3u + 1u];
    let pb = palette[i * 3u + 2u];
    let dr = pixel.x - pr;
    let dg = pixel.y - pg;
    let db = pixel.z - pb;
    let d = dr * dr + dg * dg + db * db;
    if (d < best_d) {
      best_d = d;
      best_r = pr;
      best_g = pg;
      best_b = pb;
    }
  }
  output[idx] = vec4<f32>(best_r, best_g, best_b, pixel.w);
}
"#;

impl GpuFilter for Quantize {
    fn shader_body(&self) -> &str { QUANTIZE_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.max_colors.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Palette is image-dependent — computed on CPU before GPU dispatch.
        // The pipeline pre-computes the palette and binds it here.
        // For now, return empty — the executor must call compute() first
        // to build the palette, then bind it.
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_reduces_colors() {
        let mut input = Vec::new();
        for i in 0..100 {
            let v = i as f32 / 99.0;
            input.extend_from_slice(&[v, v * 0.5, 1.0 - v, 1.0]);
        }
        let f = Quantize { max_colors: 4 };
        let out = f.compute(&input, 10, 10).unwrap();
        // Count unique colors
        let mut unique = std::collections::HashSet::new();
        for pixel in out.chunks_exact(4) {
            let key = (
                (pixel[0] * 1000.0) as i32,
                (pixel[1] * 1000.0) as i32,
                (pixel[2] * 1000.0) as i32,
            );
            unique.insert(key);
        }
        assert!(unique.len() <= 4, "Should have <=4 colors, got {}", unique.len());
    }
}
