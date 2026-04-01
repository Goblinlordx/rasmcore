//! Filter: blend_if (category: composite)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Conditionally blend two images based on luminosity ranges.
///
/// Photoshop Blend-If equivalent: pixels are blended based on their
/// luminosity. The "this layer" range controls where the top layer is
/// visible; the "underlying" range controls where the bottom layer
/// shows through. Feather creates smooth transitions at range boundaries.

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Blend-If — Photoshop-style conditional compositing by luminosity.
pub struct BlendIfParams {
    /// This layer: black point (shadows become transparent below this)
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub this_black: u32,
    /// This layer: white point (highlights become transparent above this)
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub this_white: u32,
    /// Underlying layer: black point (underlying shows through below this)
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub under_black: u32,
    /// Underlying layer: white point (underlying shows through above this)
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub under_white: u32,
    /// Feather width in luminosity levels for smooth transitions
    #[param(min = 0, max = 50, step = 1, default = 10)]
    pub feather: u32,
}

#[rasmcore_macros::register_compositor(
    name = "blend_if",
    category = "composite",
    group = "composite",
    variant = "blend_if",
    reference = "Photoshop Blend-If luminosity-range blending"
)]
pub fn blend_if(
    pixels: &[u8],
    info: &ImageInfo,
    under_data: &[u8],
    this_black: u32,
    this_white: u32,
    under_black: u32,
    under_white: u32,
    feather: u32,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Rgb8 && info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "blend_if requires RGB8 or RGBA8".into(),
        ));
    }

    let bpp = if info.format == PixelFormat::Rgba8 {
        4
    } else {
        3
    };
    let npixels = (info.width * info.height) as usize;

    // Validate underlying data matches
    if under_data.len() < npixels * bpp {
        return Err(ImageError::InvalidInput(
            "underlying image data too short for blend_if".into(),
        ));
    }

    let tb = this_black.min(255) as f32;
    let tw = this_white.min(255) as f32;
    let ub = under_black.min(255) as f32;
    let uw = under_white.min(255) as f32;
    let f = feather.min(50) as f32;

    let mut result = pixels.to_vec();

    for i in 0..npixels {
        let base = i * bpp;

        // Compute luminosity of this layer pixel (BT.709)
        let this_luma = 0.2126 * pixels[base] as f32
            + 0.7152 * pixels[base + 1] as f32
            + 0.0722 * pixels[base + 2] as f32;

        // Compute luminosity of underlying pixel
        let under_luma = 0.2126 * under_data[base] as f32
            + 0.7152 * under_data[base + 1] as f32
            + 0.0722 * under_data[base + 2] as f32;

        // This layer visibility: visible between this_black and this_white
        let this_factor = blend_if_smoothstep(this_luma, tb - f, tb + f)
            * (1.0 - blend_if_smoothstep(this_luma, tw - f, tw + f));

        // Underlying visibility: underlying shows through outside under range
        let under_factor = blend_if_smoothstep(under_luma, ub - f, ub + f)
            * (1.0 - blend_if_smoothstep(under_luma, uw - f, uw + f));

        // Combined: this layer visible where both factors are high
        let blend = (this_factor * under_factor).clamp(0.0, 1.0);
        let inv = 1.0 - blend;

        result[base] = (pixels[base] as f32 * blend + under_data[base] as f32 * inv + 0.5) as u8;
        result[base + 1] =
            (pixels[base + 1] as f32 * blend + under_data[base + 1] as f32 * inv + 0.5) as u8;
        result[base + 2] =
            (pixels[base + 2] as f32 * blend + under_data[base + 2] as f32 * inv + 0.5) as u8;
        if bpp == 4 {
            result[base + 3] = pixels[base + 3]; // preserve alpha
        }
    }

    Ok(result)
}
