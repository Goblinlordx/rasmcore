use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::{median_cut_palette, nearest_color, bayer_matrix};

/// Ordered dithering with Bayer matrix.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "dither_ordered", category = "color", cost = "O(n)")]
pub struct DitherOrdered {
    /// Max palette size (2-256).
    #[param(min = 2, max = 256, default = 16)]
    pub max_colors: u32,
    /// Bayer matrix size (2, 4, 8, or 16).
    #[param(min = 2, max = 16, default = 4)]
    pub map_size: u32,
}

impl Filter for DitherOrdered {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let palette = median_cut_palette(input, self.max_colors as usize);
        let bayer = bayer_matrix(self.map_size);
        let bayer_n = self.map_size as usize;
        let spread = 1.0 / self.max_colors as f32;
        let mut out = input.to_vec();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize * 4;
                let threshold = bayer[(y as usize % bayer_n) * bayer_n + (x as usize % bayer_n)];
                let bias = (threshold - 0.5) * spread;
                let r = out[idx] + bias;
                let g = out[idx + 1] + bias;
                let b = out[idx + 2] + bias;
                let (nr, ng, nb) = nearest_color(&palette, r, g, b);
                out[idx] = nr;
                out[idx + 1] = ng;
                out[idx + 2] = nb;
            }
        }
        Ok(out)
    }
}

/// DitherOrdered GPU — Bayer threshold + palette lookup via extra buffers.
const DITHER_ORDERED_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  map_size: u32,
  palette_size: u32,
  spread: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> bayer: array<f32>;
@group(0) @binding(4) var<storage, read> palette: array<f32>;

fn nearest_palette(r: f32, g: f32, b: f32) -> vec3<f32> {
  var best_d = 1e30;
  var best = vec3<f32>(0.0, 0.0, 0.0);
  for (var i = 0u; i < params.palette_size; i = i + 1u) {
    let pr = palette[i * 3u];
    let pg = palette[i * 3u + 1u];
    let pb = palette[i * 3u + 2u];
    let dr = r - pr;
    let dg = g - pg;
    let db = b - pb;
    let d = dr * dr + dg * dg + db * db;
    if (d < best_d) {
      best_d = d;
      best = vec3<f32>(pr, pg, pb);
    }
  }
  return best;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let bx = gid.x % params.map_size;
  let by = gid.y % params.map_size;
  let threshold = bayer[by * params.map_size + bx];
  let bias = (threshold - 0.5) * params.spread;
  let c = nearest_palette(pixel.x + bias, pixel.y + bias, pixel.z + bias);
  store_pixel(idx, vec4<f32>(c.x, c.y, c.z, pixel.w));
}
"#;

impl GpuFilter for DitherOrdered {
    fn shader_body(&self) -> &str { DITHER_ORDERED_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.map_size.to_le_bytes());
        // palette_size filled at runtime — but we need the palette first
        // (chicken-egg: palette depends on image data, not available at shader build time)
        // We pass 0 here; the pipeline should pre-compute and bind the palette buffer.
        buf.extend_from_slice(&0u32.to_le_bytes()); // placeholder
        buf.extend_from_slice(&(1.0f32 / self.max_colors as f32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Bayer matrix as f32 buffer
        let bm = bayer_matrix(self.map_size);
        let mut bayer_buf = Vec::with_capacity(bm.len() * 4);
        for v in &bm {
            bayer_buf.extend_from_slice(&v.to_le_bytes());
        }
        // Palette buffer is empty — requires image data to compute, filled by 2-pass pipeline
        vec![bayer_buf]
    }
}
