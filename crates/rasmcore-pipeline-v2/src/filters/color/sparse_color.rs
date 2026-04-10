use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

/// Sparse color interpolation — Shepard (inverse-distance weighted) method.
#[derive(Clone)]
pub struct SparseColor {
    /// Control points: (x, y, r, g, b).
    pub points: Vec<(f32, f32, f32, f32, f32)>,
    /// Inverse-distance power (default 2.0).
    pub power: f32,
}

impl Filter for SparseColor {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.points.is_empty() {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let power = self.power;
        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let px = x as f32;
                let py = y as f32;
                let mut weight_sum = 0.0f32;
                let mut r_sum = 0.0f32;
                let mut g_sum = 0.0f32;
                let mut b_sum = 0.0f32;
                let mut exact = None;
                for &(cx, cy, cr, cg, cb) in &self.points {
                    let dx = px - cx;
                    let dy = py - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < 0.001 {
                        exact = Some((cr, cg, cb));
                        break;
                    }
                    let w = 1.0 / dist.powf(power);
                    weight_sum += w;
                    r_sum += w * cr;
                    g_sum += w * cg;
                    b_sum += w * cb;
                }
                if let Some((r, g, b)) = exact {
                    out[idx] = r;
                    out[idx + 1] = g;
                    out[idx + 2] = b;
                } else if weight_sum > 1e-10 {
                    out[idx] = r_sum / weight_sum;
                    out[idx + 1] = g_sum / weight_sum;
                    out[idx + 2] = b_sum / weight_sum;
                }
            }
        }
        Ok(out)
    }
}

/// SparseColor GPU — inverse-distance weighted interpolation from control points.
const SPARSE_COLOR_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  num_points: u32,
  power: f32,
}
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> points: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let px = f32(gid.x);
  let py = f32(gid.y);
  var weight_sum = 0.0;
  var r_sum = 0.0;
  var g_sum = 0.0;
  var b_sum = 0.0;
  var exact = false;
  var exact_r = 0.0;
  var exact_g = 0.0;
  var exact_b = 0.0;
  for (var i = 0u; i < params.num_points; i = i + 1u) {
    let cx = points[i * 5u];
    let cy = points[i * 5u + 1u];
    let cr = points[i * 5u + 2u];
    let cg = points[i * 5u + 3u];
    let cb = points[i * 5u + 4u];
    let dx = px - cx;
    let dy = py - cy;
    let dist = sqrt(dx * dx + dy * dy);
    if (dist < 0.001) {
      exact = true;
      exact_r = cr;
      exact_g = cg;
      exact_b = cb;
      break;
    }
    let w = 1.0 / pow(dist, params.power);
    weight_sum += w;
    r_sum += w * cr;
    g_sum += w * cg;
    b_sum += w * cb;
  }
  if (exact) {
    store_pixel(idx, vec4<f32>(exact_r, exact_g, exact_b, pixel.w));
  } else if (weight_sum > 1e-10) {
    store_pixel(idx, vec4<f32>(r_sum / weight_sum, g_sum / weight_sum, b_sum / weight_sum, pixel.w));
  } else {
    store_pixel(idx, pixel);
  }
}
"#;

impl GpuFilter for SparseColor {
    fn shader_body(&self) -> &str {
        SPARSE_COLOR_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&(self.points.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.power.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        let mut buf = Vec::with_capacity(self.points.len() * 5 * 4);
        for &(x, y, r, g, b) in &self.points {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
            buf.extend_from_slice(&g.to_le_bytes());
            buf.extend_from_slice(&b.to_le_bytes());
        }
        vec![buf]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn sparse_color_exact_point() {
        let points = vec![(0.0, 0.0, 1.0, 0.0, 0.0)]; // top-left = red
        let f = SparseColor { points, power: 2.0 };
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (1.0, 0.0, 0.0), 0.01, "sparse exact point");
    }
}
