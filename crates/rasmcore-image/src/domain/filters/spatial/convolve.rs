//! Filter: convolve (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "convolve",
    category = "spatial",
    reference = "custom NxN kernel convolution"
)]
pub fn convolve(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    kernel: &[f32],
    config: &ConvolveParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = config.kw.max(config.kh) / 2;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let kw = config.kw;
    let kh = config.kh;
    let divisor = config.divisor;

    let kw = kw as usize;
    let kh = kh as usize;
    if kw.is_multiple_of(2) || kh.is_multiple_of(2) || kw * kh != kernel.len() {
        return Err(ImageError::InvalidParameters(
            "kernel dimensions must be odd and match kernel length".into(),
        ));
    }
    validate_format(info.format)?;

    // 16-bit: process in f32 domain, then convert back
    if is_16bit(info.format) {
        let result = process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            convolve(r, &mut u, i8, kernel, config)
        })?;
        return Ok(crop_to_request(&result, expanded, request, info.format));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;

    // Try separable path first (O(2K) vs O(K^2))
    if let Some((row_k, col_k)) = is_separable(kernel, kw, kh) {
        let result = convolve_separable(pixels, w, h, channels, &row_k, &col_k, divisor)?;
        return Ok(crop_to_request(&result, expanded, request, info.format));
    }

    // General 2D convolution with padded input
    let rw = kw / 2;
    let rh = kh / 2;
    let inv_div = 1.0 / divisor;
    let padded = pad_reflect(pixels, w, h, channels, rw.max(rh));
    let pw = w + 2 * rw.max(rh);

    let mut out = vec![0u8; pixels.len()];
    let pad = rw.max(rh);

    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    let row_off = (y + pad - rh + ky) * pw * channels;
                    for kx in 0..kw {
                        let px_off = row_off + (x + pad - rw + kx) * channels + c;
                        sum += kernel[ky * kw + kx] * padded[px_off] as f32;
                    }
                }
                out[(y * w + x) * channels + c] = (sum * inv_div).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(crop_to_request(&out, expanded, request, info.format))
}
