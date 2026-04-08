//! Drawing operation filters — SDF-based shape rendering.
//!
//! Each filter composites a shape onto the input image using signed distance
//! fields for anti-aliased edges. GPU shaders compute the SDF per-pixel.

mod draw_line;
mod draw_rect;
mod draw_circle;
mod draw_ellipse;
mod draw_arc;
mod draw_polygon;
mod solid_fill;
mod draw_text;

pub use draw_line::DrawLine;
pub use draw_rect::DrawRect;
pub use draw_circle::DrawCircle;
pub use draw_ellipse::DrawEllipse;
pub use draw_arc::DrawArc;
pub use draw_polygon::DrawPolygon;
pub use solid_fill::SolidFill;
pub use draw_text::DrawTextNode;

/// Blend a color onto a pixel with given coverage (0..1).
#[inline(always)]
fn blend(dst: &mut [f32], r: f32, g: f32, b: f32, a: f32, coverage: f32) {
    let ca = a * coverage;
    dst[0] = dst[0] * (1.0 - ca) + r * ca;
    dst[1] = dst[1] * (1.0 - ca) + g * ca;
    dst[2] = dst[2] * (1.0 - ca) + b * ca;
    dst[3] = dst[3] * (1.0 - ca) + ca;
}

/// SDF anti-aliasing: smoothstep from stroke_width to stroke_width+1 pixel.
#[inline(always)]
fn sdf_coverage(dist: f32, stroke_width: f32, fill: bool) -> f32 {
    if fill {
        smoothstep_f32(0.5, -0.5, dist)
    } else {
        let half = stroke_width * 0.5;
        smoothstep_f32(half + 0.5, half - 0.5, dist.abs())
    }
}

#[inline(always)]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Common SDF blending WGSL snippet.
const SDF_BLEND_WGSL: &str = r#"
fn sdf_coverage_fill(d: f32) -> f32 { return smoothstep(0.5, -0.5, d); }
fn sdf_coverage_stroke(d: f32, half_w: f32) -> f32 { return smoothstep(half_w + 0.5, half_w - 0.5, abs(d)); }
fn sdf_blend(bg: vec4<f32>, color: vec4<f32>, coverage: f32) -> vec4<f32> {
  let ca = color.w * coverage;
  return vec4<f32>(bg.rgb * (1.0 - ca) + color.rgb * ca, bg.w * (1.0 - ca) + ca);
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

    #[test]
    fn all_draw_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["draw_line", "draw_rect", "draw_circle", "draw_ellipse",
                       "draw_arc", "draw_polygon", "solid_fill"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn draw_circle_modifies_pixels() {
        let input = vec![0.0f32; 100 * 100 * 4];
        let f = DrawCircle {
            cx: 50.0, cy: 50.0, radius: 20.0, stroke_width: 0.0,
            color_r: 1.0, color_g: 0.0, color_b: 0.0, color_a: 1.0,
        };
        let out = f.compute(&input, 100, 100).unwrap();
        // Center pixel should be red
        let i = (50 * 100 + 50) * 4;
        assert!(out[i] > 0.9, "center R should be ~1.0, got {}", out[i]);
        // Corner pixel should be unchanged
        assert!(out[0] < 0.01, "corner should be black");
    }

    #[test]
    fn solid_fill_blends() {
        let input = vec![1.0, 1.0, 1.0, 1.0]; // white pixel
        let f = SolidFill { color_r: 0.0, color_g: 0.0, color_b: 0.0, color_a: 0.5 };
        let out = f.compute(&input, 1, 1).unwrap();
        // 50% black over white = 0.5 gray
        assert!((out[0] - 0.5).abs() < 0.01);
    }
}
