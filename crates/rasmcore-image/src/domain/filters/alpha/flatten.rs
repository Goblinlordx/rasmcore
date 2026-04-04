//! Filter: flatten (category: alpha)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Flatten RGBA to RGB by compositing onto a solid background color.
/// Registered as mapper because it changes pixel format (RGBA8 → RGB8).
/// Parameters for flatten (alpha compositing onto background).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct FlattenParams {
    /// Background red component
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub bg_r: u8,
    /// Background green component
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub bg_g: u8,
    /// Background blue component
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub bg_b: u8,
}

impl rasmcore_pipeline::GpuCapable for FlattenParams {
    fn gpu_ops(&self, _width: u32, _height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        None // Format change (RGBA8 -> RGB8) not supported on GPU
    }
}

#[rasmcore_macros::register_mapper(
    name = "flatten",
    category = "alpha",
    group = "alpha",
    variant = "flatten",
    reference = "composite onto background color",
    output_format = "Rgb8"
)]
pub fn flatten_registered(
    pixels: &[u8],
    info: &ImageInfo,
    config: &FlattenParams,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let bg_r = config.bg_r;
    let bg_g = config.bg_g;
    let bg_b = config.bg_b;

    flatten(pixels, info, [bg_r, bg_g, bg_b])
}
