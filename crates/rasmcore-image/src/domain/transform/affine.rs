use super::super::error::ImageError;
use super::super::types::{DecodedImage, ImageInfo};
use super::{bytes_per_pixel, validate_pixel_buffer};

/// Apply a general 2D affine transform.
///
/// `matrix` is [a, b, tx, c, d, ty] representing:
///   x' = a*x + b*y + tx
///   y' = c*x + d*y + ty
///
/// Output dimensions must be specified. Uses bilinear interpolation.
pub fn affine(
    pixels: &[u8],
    info: &ImageInfo,
    matrix: &[f64; 6],
    out_width: u32,
    out_height: u32,
    bg_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    let out_w = out_width as usize;
    let out_h = out_height as usize;

    // Compute inverse matrix for inverse mapping
    let [a, b, tx, c, d, ty] = *matrix;
    let det = a * d - b * c;
    if det.abs() < 1e-10 {
        return Err(ImageError::InvalidParameters(
            "singular affine matrix".into(),
        ));
    }
    let inv_det = 1.0 / det;
    let ia = d * inv_det;
    let ib = -b * inv_det;
    let ic = -c * inv_det;
    let id = a * inv_det;
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);

    let mut output = vec![0u8; out_w * out_h * bpp];

    // Vectorized bilinear interpolation: process all channels per pixel as f32x4
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let zero_v = f32x4_splat(0.0);
        let max_v = f32x4_splat(255.0);
        let half_v = f32x4_splat(0.5);
        let bg_v = f32x4(
            bg_color.first().copied().unwrap_or(0) as f32,
            bg_color.get(1).copied().unwrap_or(0) as f32,
            bg_color.get(2).copied().unwrap_or(0) as f32,
            bg_color.get(3).copied().unwrap_or(0) as f32,
        );

        for oy in 0..out_h {
            // Pre-compute row-constant part of inverse mapping
            let row_sx_base = ib * oy as f64 + itx;
            let row_sy_base = id * oy as f64 + ity;

            for ox in 0..out_w {
                let sx = ia * ox as f64 + row_sx_base;
                let sy = ic * ox as f64 + row_sy_base;
                let out_idx = (oy * out_w + ox) * bpp;

                if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    let fx = (sx - x0 as f64) as f32;
                    let fy = (sy - y0 as f64) as f32;

                    // Compute weights
                    let w00 = f32x4_splat((1.0 - fx) * (1.0 - fy));
                    let w10 = f32x4_splat(fx * (1.0 - fy));
                    let w01 = f32x4_splat((1.0 - fx) * fy);
                    let w11 = f32x4_splat(fx * fy);

                    // Load 4 source pixels (up to 4 channels each)
                    let i00 = y0 * w * bpp + x0 * bpp;
                    let i10 = y0 * w * bpp + x1 * bpp;
                    let i01 = y1 * w * bpp + x0 * bpp;
                    let i11 = y1 * w * bpp + x1 * bpp;

                    let load_px = |idx: usize| -> v128 {
                        f32x4(
                            pixels[idx] as f32,
                            if bpp > 1 { pixels[idx + 1] as f32 } else { 0.0 },
                            if bpp > 2 { pixels[idx + 2] as f32 } else { 0.0 },
                            if bpp > 3 { pixels[idx + 3] as f32 } else { 0.0 },
                        )
                    };

                    let p00 = load_px(i00);
                    let p10 = load_px(i10);
                    let p01 = load_px(i01);
                    let p11 = load_px(i11);

                    // Bilinear blend: val = p00*w00 + p10*w10 + p01*w01 + p11*w11
                    let val = f32x4_add(
                        f32x4_add(f32x4_mul(p00, w00), f32x4_mul(p10, w10)),
                        f32x4_add(f32x4_mul(p01, w01), f32x4_mul(p11, w11)),
                    );

                    // Round, clamp, store
                    let out_v = f32x4_min(max_v, f32x4_max(zero_v, f32x4_add(val, half_v)));
                    output[out_idx] = f32x4_extract_lane::<0>(out_v) as u8;
                    if bpp > 1 {
                        output[out_idx + 1] = f32x4_extract_lane::<1>(out_v) as u8;
                    }
                    if bpp > 2 {
                        output[out_idx + 2] = f32x4_extract_lane::<2>(out_v) as u8;
                    }
                    if bpp > 3 {
                        output[out_idx + 3] = f32x4_extract_lane::<3>(out_v) as u8;
                    }
                } else {
                    output[out_idx] = f32x4_extract_lane::<0>(bg_v) as u8;
                    if bpp > 1 {
                        output[out_idx + 1] = f32x4_extract_lane::<1>(bg_v) as u8;
                    }
                    if bpp > 2 {
                        output[out_idx + 2] = f32x4_extract_lane::<2>(bg_v) as u8;
                    }
                    if bpp > 3 {
                        output[out_idx + 3] = f32x4_extract_lane::<3>(bg_v) as u8;
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for oy in 0..out_h {
            let row_sx_base = ib * oy as f64 + itx;
            let row_sy_base = id * oy as f64 + ity;

            for ox in 0..out_w {
                let sx = ia * ox as f64 + row_sx_base;
                let sy = ic * ox as f64 + row_sy_base;
                let out_idx = (oy * out_w + ox) * bpp;

                if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    let fx = sx - x0 as f64;
                    let fy = sy - y0 as f64;

                    let w00 = (1.0 - fx) * (1.0 - fy);
                    let w10 = fx * (1.0 - fy);
                    let w01 = (1.0 - fx) * fy;
                    let w11 = fx * fy;

                    for ch in 0..bpp {
                        let p00 = pixels[y0 * w * bpp + x0 * bpp + ch] as f64;
                        let p10 = pixels[y0 * w * bpp + x1 * bpp + ch] as f64;
                        let p01 = pixels[y1 * w * bpp + x0 * bpp + ch] as f64;
                        let p11 = pixels[y1 * w * bpp + x1 * bpp + ch] as f64;

                        let val = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11;
                        output[out_idx + ch] = (val + 0.5).clamp(0.0, 255.0) as u8;
                    }
                } else {
                    for ch in 0..bpp {
                        output[out_idx + ch] = bg_color.get(ch).copied().unwrap_or(0);
                    }
                }
            }
        }
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: out_width,
            height: out_height,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}
