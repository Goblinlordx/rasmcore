//! Filter: premultiply (category: alpha)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Convert straight alpha to premultiplied alpha (RGBA8 only).
#[rasmcore_macros::register_filter(
    name = "premultiply",
    category = "alpha",
    group = "alpha",
    variant = "premultiply",
    reference = "premultiplied alpha conversion"
)]
pub fn premultiply(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    if info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "premultiply requires RGBA8".into(),
        ));
    }
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(4) {
        let a = chunk[3] as u16;
        chunk[0] = ((chunk[0] as u16 * a + 127) / 255) as u8;
        chunk[1] = ((chunk[1] as u16 * a + 127) / 255) as u8;
        chunk[2] = ((chunk[2] as u16 * a + 127) / 255) as u8;
    }
    Ok(result)
}
