//! Filter: draw_text (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw text on the image using the embedded 8x16 bitmap font.
#[rasmcore_macros::register_filter(
    name = "draw_text",
    category = "draw",
    group = "draw",
    variant = "text",
    reference = "bitmap 8x16 text rendering"
)]
pub fn draw_text_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    text: &str,
    config: &DrawTextParams,
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
    let scale = config.scale;
    let color_r = config.color.r as u32;
    let color_g = config.color.g as u32;
    let color_b = config.color.b as u32;
    let color_a = config.color.a as u32;

    let color = [color_r as u8, color_g as u8, color_b as u8, color_a as u8];
    let (result, _) = crate::domain::draw::draw_text(pixels, info, x, y, text, scale, color)?;
    Ok(result)
}
