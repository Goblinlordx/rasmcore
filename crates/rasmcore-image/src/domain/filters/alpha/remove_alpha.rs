//! Filter: remove_alpha (category: alpha)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Remove alpha channel from RGBA8, producing RGB8.
#[rasmcore_macros::register_mapper(
    name = "remove_alpha",
    category = "alpha",
    group = "alpha",
    variant = "remove",
    reference = "RGBA8 to RGB8 by dropping alpha channel",
    output_format = "Rgb8"
)]
pub fn remove_alpha(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "remove_alpha requires RGBA8 input".into(),
        ));
    }
    let npixels = (info.width * info.height) as usize;
    let mut rgb = Vec::with_capacity(npixels * 3);
    for chunk in pixels.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    Ok((
        rgb,
        ImageInfo {
            width: info.width,
            height: info.height,
            format: PixelFormat::Rgb8,
            color_space: info.color_space,
        },
    ))
}
