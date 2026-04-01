//! Filter: photo_filter (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply a photo filter (warming/cooling color overlay).
///
/// Blends a solid color over the image at the given density. When
/// preserve_luminosity is enabled, the original pixel's luminance is
/// maintained (only hue/saturation shifts). PS Photo Filter equivalent.

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Photo Filter — warming/cooling color overlay like a camera lens filter.
pub struct PhotoFilterParams {
    /// Filter color red
    #[param(min = 0, max = 255, step = 1, default = 236)]
    pub color_r: u32,
    /// Filter color green
    #[param(min = 0, max = 255, step = 1, default = 138)]
    pub color_g: u32,
    /// Filter color blue
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub color_b: u32,
    /// Filter density (0 = no effect, 100 = full color replacement)
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 25.0)]
    pub density: f32,
    /// Preserve luminosity (keep original brightness)
    #[param(min = 0, max = 1, step = 1, default = 1)]
    pub preserve_luminosity: u32,
}

#[rasmcore_macros::register_filter(name = "photo_filter", category = "color")]
pub fn photo_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &PhotoFilterParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    let color_r = config.color_r;
    let color_g = config.color_g;
    let color_b = config.color_b;
    let density = config.density;
    let preserve_luminosity = config.preserve_luminosity;

    let density = (density / 100.0).clamp(0.0, 1.0);
    if density == 0.0 {
        return Ok(pixels.to_vec());
    }

    let fr = color_r.min(255) as f32 / 255.0;
    let fg = color_g.min(255) as f32 / 255.0;
    let fb = color_b.min(255) as f32 / 255.0;
    let preserve = preserve_luminosity != 0;

    let bpp = match info.format {
        PixelFormat::Rgba8 => 4,
        PixelFormat::Rgb8 => 3,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "photo_filter requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(bpp) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;

        // Blend: lerp(original, filter_color, density)
        let mut nr = r + (fr - r) * density;
        let mut ng = g + (fg - g) * density;
        let mut nb = b + (fb - b) * density;

        if preserve {
            // Preserve original luminance (BT.709)
            let orig_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let new_luma = 0.2126 * nr + 0.7152 * ng + 0.0722 * nb;
            if new_luma > 0.0 {
                let scale = orig_luma / new_luma;
                nr = (nr * scale).clamp(0.0, 1.0);
                ng = (ng * scale).clamp(0.0, 1.0);
                nb = (nb * scale).clamp(0.0, 1.0);
            }
        }

        chunk[0] = (nr * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        chunk[1] = (ng * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        chunk[2] = (nb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    Ok(result)
}
