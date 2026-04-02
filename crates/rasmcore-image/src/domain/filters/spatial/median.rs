//! Filter: median (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply median filter with given radius. Window is (2*radius+1)^2.
///
/// Uses histogram sliding-window (Huang algorithm) for radius > 2 giving
/// O(1) amortized per pixel. Falls back to sorting for radius <= 2 where
/// the small window makes sorting faster than histogram maintenance.

/// Parameters for median filter.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct MedianParams {
    /// Filter radius in pixels
    #[param(min = 1, max = 20, step = 1, default = 3, hint = "rc.log_slider")]
    pub radius: u32,
}

#[rasmcore_macros::register_filter(
    name = "median", gpu = "true",
    category = "spatial",
    group = "denoise",
    variant = "median",
    reference = "median rank filter"
)]
pub fn median(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &MedianParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = config.radius;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let radius = config.radius;

    if radius == 0 {
        return Ok(pixels.to_vec());
    }
    validate_format(info.format)?;

    // 16-bit: delegate to 8-bit path (histogram-based median would need 65536 bins)
    if is_16bit(info.format) {
        let result = process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            median(r, &mut u, i8, config)
        })?;
        return Ok(crop_to_request(&result, expanded, request, info.format));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

    let result = if radius <= 2 {
        median_sort(pixels, w, h, channels, radius)?
    } else {
        median_histogram(pixels, w, h, channels, radius)?
    };
    Ok(crop_to_request(&result, expanded, request, info.format))
}
