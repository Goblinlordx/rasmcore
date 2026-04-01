//! Filter: gradient_map (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Remap image luminance through a color gradient.
///
/// Computes BT.709 luminance per pixel, then interpolates the gradient
/// stops to produce an output color. Black-to-white gradient produces
/// grayscale equivalent.
#[rasmcore_macros::register_filter(
    name = "gradient_map",
    category = "color",
    reference = "luminance-to-gradient color mapping"
)]
pub fn gradient_map(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    stops: String,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    let gradient_stops = parse_gradient_stops(&stops)?;

    // Build 256-entry LUT for fast lookup
    let mut lut = [[0u8; 3]; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let t = i as f32 / 255.0;
        *entry = interpolate_gradient(&gradient_stops, t);
    }

    let bpp = match info.format {
        PixelFormat::Rgba8 => 4,
        PixelFormat::Rgb8 => 3,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "gradient_map requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(bpp) {
        // BT.709 luminance (float for accuracy)
        let luma = (0.2126 * chunk[0] as f32 + 0.7152 * chunk[1] as f32 + 0.0722 * chunk[2] as f32)
            .round()
            .clamp(0.0, 255.0) as u8;
        let color = lut[luma as usize];
        chunk[0] = color[0];
        chunk[1] = color[1];
        chunk[2] = color[2];
        // Alpha (if RGBA) preserved
    }
    Ok(result)
}
