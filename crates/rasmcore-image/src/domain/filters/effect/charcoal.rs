//! Filter: charcoal (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Charcoal sketch: Sobel edge detection → blur → invert.
/// IM's -charcoal uses a different edge detector (not Sobel) plus normalize;
/// we use Sobel which produces visually similar but numerically different
/// edge maps. The normalize step is intentionally omitted because it
/// amplifies the edge detector difference (MAE 24→239 with normalize).
/// Registered as mapper because it changes pixel format (RGB8 → Gray8).
#[rasmcore_macros::register_mapper(
    name = "charcoal",
    category = "effect",
    reference = "charcoal drawing edge effect",
    output_format = "Gray8"
)]
pub fn charcoal(
    pixels: &[u8],
    info: &ImageInfo,
    config: &CharcoalParams,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let radius = config.radius;
    let sigma = config.sigma;

    // 1. Optional pre-blur to control edge sensitivity
    let smoothed = if sigma > 0.0 {
        blur_impl(pixels, info, &BlurParams { radius: sigma })?
    } else {
        pixels.to_vec()
    };

    // 2. Edge detection via Sobel — outputs Gray8
    let edges = sobel(&smoothed, info)?;
    let gray_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };

    // 3. Post-blur to soften the edges (on the grayscale edge image)
    let blurred = if radius > 0.0 {
        blur_impl(&edges, &gray_info, &BlurParams { radius })?
    } else {
        edges
    };

    // 4. Invert to get dark lines on white background
    let result = crate::domain::point_ops::invert(&blurred, &gray_info)?;
    Ok((result, gray_info))
}
