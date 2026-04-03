//! Filter: add_alpha (category: alpha)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for add_alpha (empty — alpha value passed directly).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct AddAlphaParams {}

impl rasmcore_pipeline::GpuCapable for AddAlphaParams {
    fn gpu_ops(&self, _width: u32, _height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        None // Format change (RGB8 -> RGBA8) not supported on GPU
    }
}

/// Add alpha channel to RGB8, producing RGBA8 with given alpha value.
#[rasmcore_macros::register_mapper(
    name = "add_alpha",
    category = "alpha",
    group = "alpha",
    variant = "add",
    reference = "RGB8 to RGBA8 with uniform alpha",
    output_format = "Rgba8"
)]
pub fn add_alpha(
    pixels: &[u8],
    info: &ImageInfo,
    alpha: u8,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "add_alpha requires RGB8 input".into(),
        ));
    }
    let npixels = (info.width * info.height) as usize;
    let mut rgba = Vec::with_capacity(npixels * 4);
    for chunk in pixels.chunks_exact(3) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(alpha);
    }
    Ok((
        rgba,
        ImageInfo {
            width: info.width,
            height: info.height,
            format: PixelFormat::Rgba8,
            color_space: info.color_space,
        },
    ))
}
