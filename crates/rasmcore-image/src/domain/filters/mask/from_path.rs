//! Generator: mask_from_path (category: mask)
//!
//! Rasterize stroke points into a grayscale mask buffer.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a mask from brush/path stroke points.
///
/// Takes flat arrays of (x, y, radius, pressure) per point and rasterizes
/// circles along the path with pressure-modulated opacity.
/// Output is RGB8 with R=G=B=gray.
///
/// Parameters encoded as comma-separated strings for generator compatibility:
/// - `points_x`, `points_y`: coordinates (0..width, 0..height)
/// - `radii`, `pressures`: per-point radius and pressure (0-1)
/// - `num_points`: number of stroke points
#[rasmcore_macros::register_generator(
    name = "mask_from_path",
    category = "mask",
    group = "mask",
    variant = "path",
    reference = "brush stroke rasterization into mask"
)]
pub fn mask_from_path(
    width: u32,
    height: u32,
    _num_points: u32,
    points_x: f32,
    points_y: f32,
    radius: f32,
    pressure: f32,
) -> Vec<u8> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    // Accumulator in f32 for additive blending
    let mut accum = vec![0.0f32; w * h];

    // Single point mode — rasterize a circle at (points_x, points_y)
    rasterize_circle(&mut accum, w, h, points_x, points_y, radius, pressure);

    // Convert to RGB8 with R=G=B
    let mut pixels = vec![0u8; w * h * 3];
    for i in 0..w * h {
        let v = (accum[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        pixels[i * 3] = v;
        pixels[i * 3 + 1] = v;
        pixels[i * 3 + 2] = v;
    }
    pixels
}

/// Rasterize a soft circle into the accumulator buffer.
fn rasterize_circle(buf: &mut [f32], w: usize, h: usize, cx: f32, cy: f32, r: f32, p: f32) {
    let r = r.max(0.5);
    let r2 = r * r;
    let x0 = ((cx - r - 1.0).floor().max(0.0)) as usize;
    let y0 = ((cy - r - 1.0).floor().max(0.0)) as usize;
    let x1 = ((cx + r + 1.0).ceil() as usize).min(w);
    let y1 = ((cy + r + 1.0).ceil() as usize).min(h);

    for y in y0..y1 {
        for x in x0..x1 {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let d2 = dx * dx + dy * dy;
            if d2 < r2 {
                // Soft falloff: 1 at center, 0 at edge
                let falloff = 1.0 - (d2 / r2).sqrt();
                let opacity = falloff * p;
                // Additive blending (clamped later)
                buf[y * w + x] = (buf[y * w + x] + opacity).min(1.0);
            }
        }
    }
}

/// Rasterize a line of circles between two points (for multi-point paths).
pub fn rasterize_stroke_segment(
    buf: &mut [f32],
    w: usize,
    h: usize,
    x0: f32,
    y0: f32,
    r0: f32,
    p0: f32,
    x1: f32,
    y1: f32,
    r1: f32,
    p1: f32,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dist = (dx * dx + dy * dy).sqrt();
    let spacing = ((r0 + r1) / 2.0 * 0.25).max(1.0);
    let steps = (dist / spacing).ceil() as usize;

    for i in 0..=steps {
        let t = if steps > 0 {
            i as f32 / steps as f32
        } else {
            0.0
        };
        let cx = x0 + dx * t;
        let cy = y0 + dy * t;
        let r = r0 + (r1 - r0) * t;
        let p = p0 + (p1 - p0) * t;
        rasterize_circle(buf, w, h, cx, cy, r, p);
    }
}
