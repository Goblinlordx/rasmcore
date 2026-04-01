use super::super::error::ImageError;
use super::super::types::ImageInfo;
use super::{ensure_rgba8, make_paint, make_stroke, pixels_to_pixmap, pixmap_to_pixels};

/// Draw an ellipse on the image.
///
/// `cx`, `cy` are the center, `rx` and `ry` are the radii along the X and Y
/// axes. Uses `tiny_skia::PathBuilder::from_oval` for native-quality bezier
/// approximation.
pub fn draw_ellipse(
    pixels: &[u8],
    info: &ImageInfo,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    color: [u8; 4],
    stroke_width: f32,
    filled: bool,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if rx <= 0.0 || ry <= 0.0 {
        return Err(ImageError::InvalidParameters(format!(
            "draw_ellipse: radii must be positive (rx={rx}, ry={ry})"
        )));
    }
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let oval_rect = resvg::tiny_skia::Rect::from_xywh(cx - rx, cy - ry, rx * 2.0, ry * 2.0)
        .ok_or_else(|| {
            ImageError::InvalidParameters(format!(
                "draw_ellipse: invalid oval ({cx},{cy},{rx},{ry})"
            ))
        })?;
    let path = resvg::tiny_skia::PathBuilder::from_oval(oval_rect).ok_or_else(|| {
        ImageError::InvalidParameters("draw_ellipse: failed to build oval path".into())
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
