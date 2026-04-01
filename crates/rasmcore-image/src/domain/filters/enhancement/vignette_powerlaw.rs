//! Filter: vignette_powerlaw (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Power-law vignette — simple radial falloff.
///
/// Multiplies each pixel by `1.0 - strength * (dist / max_dist)^falloff`.
/// This is a computationally cheap alternative to the Gaussian vignette
/// with a different aesthetic (smooth polynomial falloff vs. Gaussian).
#[allow(clippy::too_many_arguments)]

/// Parameters for the power-law vignette mode.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct VignettePowerlawParams {
    /// Darkening strength (0=none, 1=fully black at corners)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
    /// Radial falloff exponent (higher = sharper transition)
    #[param(min = 0.5, max = 5.0, step = 0.1, default = 2.0)]
    pub falloff: f32,
    /// Full canvas width
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub full_width: u32,
    /// Full canvas height
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub full_height: u32,
    /// X offset
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub offset_x: u32,
    /// Y offset
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub offset_y: u32,
}

#[rasmcore_macros::register_filter(
    name = "vignette_powerlaw",
    category = "enhancement",
    group = "vignette",
    variant = "powerlaw",
    reference = "power-law radial falloff"
)]
pub fn vignette_powerlaw(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &VignettePowerlawParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let strength = config.strength;
    let falloff = config.falloff;
    let full_width = config.full_width;
    let full_height = config.full_height;
    let offset_x = config.offset_x;
    let offset_y = config.offset_y;

    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            vignette_powerlaw(r, &mut u, i8, config)
        });
    }

    let ch = channels(info.format);
    let color_ch = if ch == 4 { 3 } else { ch };
    let w = info.width as usize;
    let h = info.height as usize;
    let fw = full_width as f64;
    let fh = full_height as f64;
    let cx = fw / 2.0;
    let cy = fh / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();
    let strength_d = strength as f64;
    let falloff_d = falloff as f64;

    let mut result = pixels.to_vec();

    for row in 0..h {
        let abs_y = (offset_y as usize + row) as f64 + 0.5;
        let dy = abs_y - cy;
        let dy2 = dy * dy;
        for col in 0..w {
            let abs_x = (offset_x as usize + col) as f64 + 0.5;
            let dx = abs_x - cx;
            let dist = (dx * dx + dy2).sqrt();
            let t = (dist / max_dist).powf(falloff_d);
            let factor = 1.0 - strength_d * t;

            let idx = (row * w + col) * ch;
            for c in 0..color_ch {
                let v = result[idx + c] as f64 * factor;
                result[idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(result)
}
