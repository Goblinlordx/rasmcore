//! Filter: vignette (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Gaussian vignette effect — ImageMagick-compatible.
///
/// Darkens image edges using an anti-aliased elliptical Gaussian mask:
/// 1. Build an AA elliptical mask (8×8 supersampling at boundary pixels)
/// 2. Gaussian-blur the mask (separable, sigma controls transition softness)
/// 3. Multiply each pixel by the blurred mask value
///
/// This matches ImageMagick's `-vignette 0x{sigma}+{x_off}+{y_off}` within
/// MAE < 1.0 at 8-bit (max error ≤ 3).
///
/// `full_width`/`full_height` and `tile_offset_x`/`tile_offset_y` support
/// tiled execution. For non-tiled usage, set tile offsets to 0 and full dims
/// to the image dimensions.
#[allow(clippy::too_many_arguments)]

/// Parameters for the default (Gaussian) vignette effect.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct VignetteParams {
    /// Gaussian blur sigma controlling the softness of the transition
    #[param(
        min = 1.0,
        max = 100.0,
        step = 1.0,
        default = 20.0,
        hint = "rc.log_slider"
    )]
    pub sigma: f32,
    /// Horizontal inset from edges (pixels) where darkening begins
    #[param(min = 0, max = 4000, step = 1, default = 10, hint = "rc.pixels")]
    pub x_inset: u32,
    /// Vertical inset from edges (pixels) where darkening begins
    #[param(min = 0, max = 4000, step = 1, default = 10, hint = "rc.pixels")]
    pub y_inset: u32,
    /// Full canvas width (for tile pipeline)
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub full_width: u32,
    /// Full canvas height (for tile pipeline)
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub full_height: u32,
    /// Tile X offset (for tile pipeline)
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub tile_offset_x: u32,
    /// Tile Y offset (for tile pipeline)
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub tile_offset_y: u32,
}

#[rasmcore_macros::register_filter(
    name = "vignette",
    category = "enhancement",
    group = "vignette",
    reference = "Gaussian radial darkening"
)]
pub fn vignette(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &VignetteParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let sigma = config.sigma;
    let x_inset = config.x_inset;
    let y_inset = config.y_inset;
    let full_width = config.full_width;
    let full_height = config.full_height;
    let tile_offset_x = config.tile_offset_x;
    let tile_offset_y = config.tile_offset_y;

    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            vignette(r, &mut u, i8, config)
        });
    }

    let fw = full_width as usize;
    let fh = full_height as usize;
    let cx = fw as f64 / 2.0;
    let cy = fh as f64 / 2.0;
    let rx = (fw as f64 / 2.0 - x_inset as f64).max(1.0);
    let ry = (fh as f64 / 2.0 - y_inset as f64).max(1.0);

    // Build anti-aliased elliptical mask for the full image
    let mask = build_aa_ellipse_mask(fw, fh, cx, cy, rx, ry);

    // Gaussian blur the mask
    let blurred = gaussian_blur_mask(&mask, fw, fh, sigma as f64);

    // Multiply pixels by the corresponding mask region
    let ch = channels(info.format);
    let color_ch = if ch == 4 { 3 } else { ch };
    let tw = info.width as usize;
    let th = info.height as usize;
    let tx = tile_offset_x as usize;
    let ty = tile_offset_y as usize;

    let mut result = pixels.to_vec();
    for row in 0..th {
        for col in 0..tw {
            let factor = blurred[(ty + row) * fw + (tx + col)];
            let idx = (row * tw + col) * ch;
            for c in 0..color_ch {
                let v = result[idx + c] as f64 * factor;
                result[idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(result)
}
