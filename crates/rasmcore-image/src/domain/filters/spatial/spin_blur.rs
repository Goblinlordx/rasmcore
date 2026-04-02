//! Filter: spin_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply rotational (spin) blur around image center.
///
/// Matches ImageMagick's `-rotational-blur` algorithm exactly:
/// - Global sample count based on angle and image diagonal
/// - Uniform angular sweep for all pixels (same rotation range everywhere)
/// - Position-vector rotation (not arc sampling)
/// - Bilinear interpolation with edge clamp
///
/// center_x/center_y are normalized (0.5 = image center, matching IM default).

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Spin Blur — rotational motion blur around a center point.
pub struct SpinBlurParams {
    /// Center X as fraction of width (0.0 = left, 0.5 = center, 1.0 = right)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    /// Center Y as fraction of height (0.0 = top, 0.5 = center, 1.0 = bottom)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
    /// Rotation angle in degrees (max blur at edges)
    #[param(min = 0.0, max = 360.0, step = 0.5, default = 10.0)]
    pub angle: f32,
}

#[rasmcore_macros::register_filter(name = "spin_blur", gpu = "true", category = "spatial")]
pub fn spin_blur(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SpinBlurParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    let center_x = config.center_x;
    let center_y = config.center_y;
    let angle = config.angle;

    if angle == 0.0 {
        return Ok(pixels.to_vec());
    }

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            spin_blur(r, &mut u, i8, config)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let cx = center_x * w as f32;
    let cy = center_y * h as f32;
    let angle_rad = (angle as f64).to_radians();

    // IM algorithm: global sample count = (2 * ceil(angle_rad * diagonal/2) + 2) | 1
    let half_diag = ((w as f64 / 2.0).powi(2) + (h as f64 / 2.0).powi(2)).sqrt();
    let mut n = (2.0 * (angle_rad * half_diag).ceil() + 2.0) as usize;
    if n.is_multiple_of(2) {
        n += 1;
    }
    n = n.max(3);

    // Precompute cos/sin table: rotation offsets centered around zero
    let half_n = n / 2;
    let mut cos_table = vec![0.0f64; n];
    let mut sin_table = vec![0.0f64; n];
    for i in 0..n {
        let offset = angle_rad * (i as f64 - half_n as f64) / n as f64;
        cos_table[i] = offset.cos();
        sin_table[i] = offset.sin();
    }

    let inv_n = 1.0 / n as f64;
    let mut out = vec![0u8; w * h * ch];

    for py in 0..h {
        for px in 0..w {
            let dx = px as f64 - cx as f64;
            let dy = py as f64 - cy as f64;

            let mut accum = vec![0.0f64; ch];

            for j in 0..n {
                // Rotate (dx, dy) by the j-th angle offset
                let sx = cx as f64 + dx * cos_table[j] - dy * sin_table[j];
                let sy = cy as f64 + dx * sin_table[j] + dy * cos_table[j];

                // Bilinear interpolation with edge clamp
                let fx = sx.floor();
                let fy = sy.floor();
                let frac_x = sx - fx;
                let frac_y = sy - fy;
                let x0 = (fx as isize).clamp(0, w as isize - 1) as usize;
                let y0 = (fy as isize).clamp(0, h as isize - 1) as usize;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);

                let w00 = (1.0 - frac_x) * (1.0 - frac_y);
                let w10 = frac_x * (1.0 - frac_y);
                let w01 = (1.0 - frac_x) * frac_y;
                let w11 = frac_x * frac_y;

                let i00 = (y0 * w + x0) * ch;
                let i10 = (y0 * w + x1) * ch;
                let i01 = (y1 * w + x0) * ch;
                let i11 = (y1 * w + x1) * ch;

                for c in 0..ch {
                    accum[c] += pixels[i00 + c] as f64 * w00
                        + pixels[i10 + c] as f64 * w10
                        + pixels[i01 + c] as f64 * w01
                        + pixels[i11 + c] as f64 * w11;
                }
            }

            let dst = (py * w + px) * ch;
            for c in 0..ch {
                out[dst + c] = (accum[c] * inv_n + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(out)
}
