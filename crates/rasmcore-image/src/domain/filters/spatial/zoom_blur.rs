//! Filter: zoom_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ZoomBlurParams {
    pub center_x: f32,
    pub center_y: f32,
    pub factor: f32,
}

#[rasmcore_macros::register_filter(
    name = "zoom_blur",
    category = "spatial",
    group = "blur",
    variant = "zoom",
    reference = "radial kernel simulating lens zoom"
)]
pub fn zoom_blur(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ZoomBlurParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let center_x = config.center_x;
    let center_y = config.center_y;
    let factor = config.factor;

    validate_format(info.format)?;

    if factor == 0.0 {
        return Ok(pixels.to_vec());
    }

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            zoom_blur(r, &mut u, i8, config)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let cx = center_x * w as f32;
    let cy = center_y * h as f32;

    let mut out = vec![0u8; w * h * ch];

    for py in 0..h {
        for px in 0..w {
            // Ray endpoint: pixel + (center - pixel) * factor
            let x_start = px as f32;
            let y_start = py as f32;
            let x_end = x_start + (cx - x_start) * factor;
            let y_end = y_start + (cy - y_start) * factor;

            // Adaptive sample count: ceil(distance) + 1, min 3
            // Matches GEGL's motion-blur-zoom.c
            let dist = ((x_end - x_start).powi(2) + (y_end - y_start).powi(2)).sqrt();
            let mut xy_len = (dist.ceil() as usize) + 1;
            xy_len = xy_len.max(3);

            // Soft performance cap above 100 (GEGL behavior)
            if xy_len > 100 {
                xy_len = (100 + ((xy_len - 100) as f32).sqrt() as usize).min(200);
            }

            let inv_len = 1.0 / xy_len as f32;
            let dxx = (x_end - x_start) * inv_len;
            let dyy = (y_end - y_start) * inv_len;

            // Walk along the ray, accumulating bilinear samples
            let mut ix = x_start;
            let mut iy = y_start;
            let mut accum = vec![0.0f32; ch];

            for _ in 0..xy_len {
                // Bilinear interpolation with edge-clamp
                let fx = ix.floor();
                let fy = iy.floor();
                let dx = ix - fx;
                let dy = iy - fy;

                let x0 = (fx as i32).clamp(0, w as i32 - 1) as usize;
                let y0 = (fy as i32).clamp(0, h as i32 - 1) as usize;
                let x1 = ((fx as i32) + 1).clamp(0, w as i32 - 1) as usize;
                let y1 = ((fy as i32) + 1).clamp(0, h as i32 - 1) as usize;

                for c in 0..ch {
                    let p00 = pixels[(y0 * w + x0) * ch + c] as f32;
                    let p10 = pixels[(y0 * w + x1) * ch + c] as f32;
                    let p01 = pixels[(y1 * w + x0) * ch + c] as f32;
                    let p11 = pixels[(y1 * w + x1) * ch + c] as f32;

                    // GEGL bilinear: lerp columns, then across
                    let mix0 = dy * (p01 - p00) + p00;
                    let mix1 = dy * (p11 - p10) + p10;
                    accum[c] += dx * (mix1 - mix0) + mix0;
                }

                ix += dxx;
                iy += dyy;
            }

            // Average all samples (equal weight — box filter)
            let dst = (py * w + px) * ch;
            for c in 0..ch {
                out[dst + c] = (accum[c] * inv_len + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(out)
}
