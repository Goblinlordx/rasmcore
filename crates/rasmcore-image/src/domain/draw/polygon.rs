use super::super::error::ImageError;
use super::super::types::ImageInfo;
use super::{ensure_rgba8, make_paint, make_stroke, pixels_to_pixmap, pixmap_to_pixels};

/// Draw a polygon on the image from a list of 2D points.
///
/// `points` must have at least 3 entries. The path is closed automatically.
/// If `filled` is true, the polygon is filled. Otherwise only the outline
/// is drawn with the given `stroke_width`.
pub fn draw_polygon(
    pixels: &[u8],
    info: &ImageInfo,
    points: &[super::super::param_types::Point2D],
    fill_color: [u8; 4],
    stroke_color: [u8; 4],
    stroke_width: f32,
    filled: bool,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if points.len() < 3 {
        return Err(ImageError::InvalidParameters(format!(
            "draw_polygon: need at least 3 points, got {}",
            points.len()
        )));
    }
    let vertices: Vec<(f32, f32)> = points.iter().map(|p| (p.x, p.y)).collect();
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let mut pb = resvg::tiny_skia::PathBuilder::new();
    pb.move_to(vertices[0].0, vertices[0].1);
    for &(x, y) in &vertices[1..] {
        pb.line_to(x, y);
    }
    pb.close();
    let path = pb.finish().ok_or_else(|| {
        ImageError::InvalidParameters("draw_polygon: invalid path coordinates".into())
    })?;

    if filled {
        let paint = make_paint(fill_color[0], fill_color[1], fill_color[2], fill_color[3]);
        pixmap.fill_path(
            &path,
            &paint,
            resvg::tiny_skia::FillRule::Winding,
            resvg::tiny_skia::Transform::identity(),
            None,
        );
    }

    if stroke_width > 0.0 {
        let paint = make_paint(
            stroke_color[0],
            stroke_color[1],
            stroke_color[2],
            stroke_color[3],
        );
        let stroke = make_stroke(stroke_width);
        pixmap.stroke_path(
            &path,
            &paint,
            &stroke,
            resvg::tiny_skia::Transform::identity(),
            None,
        );
    }

    Ok((pixmap_to_pixels(&pixmap), out_info))
}
