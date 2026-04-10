//! Interactive tool filters — brush and region-based operations.
//!
//! These model single-application tool strokes as filters with fixed params.
//! Full interactive brush support (paths, pressure, accumulation) is a
//! separate architectural layer — these are the per-application primitives.

mod ca_remove;
mod clone_stamp;
mod flood_fill;
mod healing_brush;
mod red_eye_remove;
mod smudge;
mod sponge;

pub use ca_remove::CaRemove;
pub use clone_stamp::CloneStamp;
pub use flood_fill::FloodFill;
pub use healing_brush::HealingBrush;
pub use red_eye_remove::RedEyeRemove;
pub use smudge::Smudge;
pub use sponge::Sponge;

const SAMPLE_BILINEAR_WGSL: &str = r#"
fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx)); let iy = i32(floor(fy));
  let dx = fx - f32(ix); let dy = fy - f32(iy);
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

#[inline(always)]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

    #[test]
    fn all_tool_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &[
            "clone_stamp",
            "smudge",
            "sponge",
            "red_eye_remove",
            "ca_remove",
            "flood_fill",
            "healing_brush",
        ] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn flood_fill_replaces_region() {
        // 4x4 all white
        let input = vec![1.0f32; 4 * 4 * 4];
        let f = FloodFill {
            seed_x: 0.5,
            seed_y: 0.5,
            tolerance: 0.5,
            fill_r: 1.0,
            fill_g: 0.0,
            fill_b: 0.0,
        };
        let out = f.compute(&input, 4, 4).unwrap();
        // All pixels should be red (all were white, connected, within tolerance)
        assert!(out[0] > 0.9); // R
        assert!(out[1] < 0.1); // G
    }
}
