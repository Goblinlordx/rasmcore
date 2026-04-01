use super::super::error::ImageError;
use super::super::types::ImageInfo;
use super::{ensure_rgba8, make_paint, make_stroke, pixels_to_pixmap, pixmap_to_pixels};

/// Draw a circle on the image.
///
/// If `filled` is true, the circle is filled. Otherwise only the outline
/// is drawn with the given `stroke_width`.
pub fn draw_circle(
    pixels: &[u8],
    info: &ImageInfo,
    cx: f32,
    cy: f32,
    radius: f32,
    color: [u8; 4],
    stroke_width: f32,
    filled: bool,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let path = resvg::tiny_skia::PathBuilder::from_circle(cx, cy, radius).ok_or_else(|| {
        ImageError::InvalidParameters(format!(
            "draw_circle: invalid circle ({cx},{cy},r={radius})"
        ))
    })?;

    let paint = make_paint(color[0], color[1], color[2], color[3]);

    if filled {
        pixmap.fill_path(
            &path,
            &paint,
            resvg::tiny_skia::FillRule::Winding,
            resvg::tiny_skia::Transform::identity(),
            None,
        );
    } else {
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
