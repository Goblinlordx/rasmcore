//! Distortion filters — coordinate remapping with bilinear sampling.
//!
//! All filters use inverse mapping: for each output pixel, compute the
//! source coordinate, then bilinear-sample the input. GPU shaders follow
//! the same pattern with `sample_bilinear_f32()`.

mod barrel;
mod spherize;
mod swirl;
mod ripple;
mod wave;
mod polar;
mod depolar;
mod liquify;
mod mesh_warp;
mod displacement_map;

pub use barrel::Barrel;
pub use spherize::Spherize;
pub use swirl::Swirl;
pub use ripple::Ripple;
pub use wave::Wave;
pub use polar::Polar;
pub use depolar::Depolar;
pub use liquify::Liquify;
pub use mesh_warp::MeshWarp;
pub use displacement_map::DisplacementMap;

/// WGSL bilinear sampling helper — prepended to each distortion shader.
const SAMPLE_BILINEAR_WGSL: &str = r#"
fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let dx = fx - f32(ix);
  let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.width) - 1);
  let y0 = clamp(iy, 0, i32(params.height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.height) - 1);
  let p00 = input[u32(x0) + u32(y0) * params.width];
  let p10 = input[u32(x1) + u32(y0) * params.width];
  let p01 = input[u32(x0) + u32(y1) * params.width];
  let p11 = input[u32(x1) + u32(y1) * params.width];
  return mix(mix(p00, p10, vec4<f32>(dx)), mix(p01, p11, vec4<f32>(dx)), vec4<f32>(dy));
}
"#;

// ─── GPU param helpers ─────────────────────────────────────────────────────

fn gpu_params_push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn gpu_params_push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

    fn solid_image(w: u32, h: u32, color: [f32; 4]) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut pixels = Vec::with_capacity(n * 4);
        for _ in 0..n {
            pixels.extend_from_slice(&color);
        }
        pixels
    }

    #[test]
    fn barrel_identity_at_zero() {
        let input = solid_image(4, 4, [0.5, 0.3, 0.1, 1.0]);
        let f = Barrel { k1: 0.0, k2: 0.0 };
        let out = f.compute(&input, 4, 4).unwrap();
        // At k1=k2=0, distortion factor = 1.0, so output ≈ input
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.02, "expected {a}, got {b}");
        }
    }

    #[test]
    fn swirl_identity_at_zero_angle() {
        let input = solid_image(4, 4, [0.5, 0.3, 0.1, 1.0]);
        let f = Swirl { angle: 0.0, radius: 100.0 };
        let out = f.compute(&input, 4, 4).unwrap();
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.02, "expected {a}, got {b}");
        }
    }

    #[test]
    fn wave_changes_pixels() {
        let mut input = vec![0.0f32; 16 * 16 * 4];
        for i in 0..input.len() { input[i] = (i as f32 / input.len() as f32); }
        let f = Wave { amplitude: 5.0, wavelength: 8.0, vertical: false };
        let out = f.compute(&input, 16, 16).unwrap();
        assert_ne!(input, out);
    }

    #[test]
    fn all_distortion_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["barrel", "spherize", "swirl", "ripple", "wave",
                       "polar", "depolar", "liquify", "mesh_warp", "displacement_map"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }
}
