//! Filter: draw_text_ttf (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw TrueType/OpenType text on the image.
///
/// Font data is passed as bytes (no filesystem access — WASM-compatible).
/// Falls back to bitmap font when font_data is empty.
#[rasmcore_macros::register_filter(name = "draw_text_ttf", category = "draw")]
pub fn draw_text_ttf_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    text: &str,
    config: &DrawTextTtfParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let x = config.x;
    let y = config.y;
    let font_size_pt = config.font_size_pt;
    let color = [
        config.color_r as u8,
        config.color_g as u8,
        config.color_b as u8,
        config.color_a as u8,
    ];
    // Without font data (scalar params only), fall back to bitmap
    let scale = (font_size_pt / 12.0).round().max(1.0) as u32;
    let (result, _) = crate::domain::draw::draw_text(pixels, info, x, y, text, scale, color)?;
    Ok(result)
}
