use super::super::error::ImageError;
use super::super::types::ImageInfo;
use super::{ensure_rgba8, make_paint, make_stroke, pixels_to_pixmap, pixmap_to_pixels};

/// Draw a line on the image.
///
/// Coordinates are in pixels. Color is RGBA. Line width in pixels.
pub fn draw_line(
    pixels: &[u8],
    info: &ImageInfo,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: [u8; 4],
    width: f32,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let mut pb = resvg::tiny_skia::PathBuilder::new();
    pb.move_to(x1, y1);
    pb.line_to(x2, y2);
    let path = pb.finish().ok_or_else(|| {
        ImageError::InvalidParameters("draw_line: invalid path coordinates".into())
    })?;

    let paint = make_paint(color[0], color[1], color[2], color[3]);
    let stroke = make_stroke(width);

    pixmap.stroke_path(
        &path,
        &paint,
        &stroke,
        resvg::tiny_skia::Transform::identity(),
        None,
    );

    Ok((pixmap_to_pixels(&pixmap), out_info))
}
