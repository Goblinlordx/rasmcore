use super::super::error::ImageError;
use super::super::types::ImageInfo;
use super::{ensure_rgba8, make_paint, make_stroke, pixels_to_pixmap, pixmap_to_pixels};

/// Draw an arc (partial ellipse outline) on the image.
///
/// `cx`, `cy` are the center, `rx` and `ry` are the radii.
/// `start_angle` and `end_angle` are in degrees (0 = right, counter-clockwise).
/// The arc is always stroked (not filled).
///
/// Uses cubic bezier segments to approximate each quadrant of the arc.
pub fn draw_arc(
    pixels: &[u8],
    info: &ImageInfo,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    start_angle: f32,
    end_angle: f32,
    color: [u8; 4],
    stroke_width: f32,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if rx <= 0.0 || ry <= 0.0 {
        return Err(ImageError::InvalidParameters(format!(
            "draw_arc: radii must be positive (rx={rx}, ry={ry})"
        )));
    }
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let path = build_arc_path(cx, cy, rx, ry, start_angle, end_angle).ok_or_else(|| {
        ImageError::InvalidParameters("draw_arc: failed to build arc path".into())
    })?;

    let paint = make_paint(color[0], color[1], color[2], color[3]);
    let stroke = make_stroke(stroke_width);
    pixmap.stroke_path(
        &path,
        &paint,
        &stroke,
        resvg::tiny_skia::Transform::identity(),
        None,
    );

    Ok((pixmap_to_pixels(&pixmap), out_info))
}

/// Build a cubic bezier arc path. Splits the angular range into segments of
/// at most 90 degrees each, using the standard bezier arc approximation.
fn build_arc_path(
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    start_deg: f32,
    end_deg: f32,
) -> Option<resvg::tiny_skia::Path> {
    let mut pb = resvg::tiny_skia::PathBuilder::new();

    let start_rad = start_deg.to_radians();
    let end_rad = end_deg.to_radians();

    // Normalize: ensure we sweep in the positive direction
    let mut sweep = end_rad - start_rad;
    if sweep.abs() < 1e-6 {
        return None;
    }
    if sweep < 0.0 {
        sweep += std::f32::consts::TAU;
    }

    // Split into segments of at most PI/2 (90 degrees)
    let max_segment = std::f32::consts::FRAC_PI_2;
    let n_segments = (sweep / max_segment).ceil() as usize;
    let segment_angle = sweep / n_segments as f32;

    // Start point
    let sx = cx + rx * start_rad.cos();
    let sy = cy - ry * start_rad.sin();
    pb.move_to(sx, sy);

    let mut angle = start_rad;
    for _ in 0..n_segments {
        let next_angle = angle + segment_angle;
        arc_bezier_segment(&mut pb, cx, cy, rx, ry, angle, next_angle);
        angle = next_angle;
    }

    pb.finish()
}

/// Append a single cubic bezier segment approximating an elliptical arc.
///
/// Uses the standard parametric bezier approximation for circular arcs
/// scaled by (rx, ry). The magic factor `alpha = 4/3 * tan(da/4)` ensures
/// the bezier curve passes through the arc endpoints with correct tangents.
fn arc_bezier_segment(
    pb: &mut resvg::tiny_skia::PathBuilder,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    a1: f32,
    a2: f32,
) {
    let da = a2 - a1;
    let alpha = (da / 4.0).tan() * 4.0 / 3.0;

    let cos1 = a1.cos();
    let sin1 = a1.sin();
    let cos2 = a2.cos();
    let sin2 = a2.sin();

    // Control point 1: tangent at start
    let cp1x = cx + rx * (cos1 - alpha * sin1);
    let cp1y = cy - ry * (sin1 + alpha * cos1);

    // Control point 2: tangent at end
    let cp2x = cx + rx * (cos2 + alpha * sin2);
    let cp2y = cy - ry * (sin2 - alpha * cos2);

    // End point
    let ex = cx + rx * cos2;
    let ey = cy - ry * sin2;

    pb.cubic_to(cp1x, cp1y, cp2x, cp2y, ex, ey);
}
