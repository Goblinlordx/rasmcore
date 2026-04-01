//! Filter: pyramid_detail_remap (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Pyramid detail remapping — edge-aware detail enhancement/smoothing.
///
/// Decomposes the image into a Gaussian/Laplacian pyramid and remaps
/// detail coefficients at each level via a sigmoidal curve:
/// `f(d) = d * sigma / (sigma + |d|)`.
///
/// - `sigma < 1.0`: compresses large gradients, enhances fine detail
/// - `sigma = 1.0`: near-identity (slight compression at large gradients)
/// - `sigma > 1.0`: suppresses fine detail (smoothing)
///
/// This is a Laplacian pyramid coefficient remapping filter, distinct from
/// the Paris et al. 2011 "Local Laplacian Filter" which rebuilds the pyramid
/// per-pixel with a power-law remapping.
///
/// - `sigma`: detail remapping strength (0.2 = strong enhancement, 1.0 = neutral, 3.0 = smooth)
/// - `num_levels`: pyramid depth (0 = auto, typically 5-7)
#[rasmcore_macros::register_filter(
    name = "pyramid_detail_remap",
    category = "enhancement",
    reference = "Laplacian pyramid detail enhancement"
)]
pub fn pyramid_detail_remap(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &PyramidDetailRemapParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let sigma = config.sigma;
    let num_levels = config.num_levels;

    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "pyramid_detail_remap requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let (w, h) = (info.width as usize, info.height as usize);

    // Determine pyramid levels
    let levels = if num_levels == 0 {
        ((w.min(h) as f32).log2() as usize).clamp(2, 7)
    } else {
        (num_levels as usize).min(10)
    };

    // Process each channel independently through the pyramid
    let mut result = vec![0u8; pixels.len()];

    for c in 0..3 {
        // Extract single channel as f32
        let channel: Vec<f32> = (0..w * h)
            .map(|i| pixels[i * channels + c] as f32 / 255.0)
            .collect();

        let output = pyramid_detail_remap_channel(&channel, w, h, levels, sigma);

        // Write back
        for i in 0..w * h {
            result[i * channels + c] = (output[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    // Copy alpha if present
    if channels == 4 {
        for i in 0..w * h {
            result[i * 4 + 3] = pixels[i * 4 + 3];
        }
    }

    Ok(result)
}
