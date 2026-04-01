//! Filter: perspective_warp (category: advanced)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "perspective_warp",
    category = "advanced",
    group = "perspective",
    variant = "warp",
    reference = "3x3 homography transformation"
)]
pub fn perspective_warp(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    matrix: &[f64],
    config: &PerspectiveWarpParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let out_width = config.out_width;
    let out_height = config.out_height;

    if matrix.len() != 9 {
        return Err(ImageError::InvalidParameters(format!(
            "perspective_warp requires 9-element matrix, got {}",
            matrix.len()
        )));
    }
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            perspective_warp(r, &mut u, i8, matrix, config)
        });
    }

    let in_w = info.width as i32;
    let in_h = info.height as i32;
    let ow = out_width as usize;
    let oh = out_height as usize;
    let ch = channels(info.format);

    let wtab = build_bilinear_tab();
    let mut out = vec![0u8; ow * oh * ch];
    let tab_sz = INTER_TAB_SIZE as f64;

    for oy in 0..oh {
        // Per-row base values
        let base_x = matrix[1] * oy as f64 + matrix[2];
        let base_y = matrix[4] * oy as f64 + matrix[5];
        let base_w = matrix[7] * oy as f64 + matrix[8];

        for ox in 0..ow {
            let w = base_w + matrix[6] * ox as f64;
            let w_scaled = if w != 0.0 { tab_sz / w } else { 0.0 };

            let fx = ((base_x + matrix[0] * ox as f64) * w_scaled)
                .max(i32::MIN as f64)
                .min(i32::MAX as f64);
            let fy = ((base_y + matrix[3] * ox as f64) * w_scaled)
                .max(i32::MIN as f64)
                .min(i32::MAX as f64);

            let ix = cv_round(fx);
            let iy = cv_round(fy);

            // Integer source coords (right-shift by INTER_BITS = 5)
            let sx = ix >> INTER_BITS;
            let sy = iy >> INTER_BITS;

            // Sub-pixel index into weight table
            let alpha = ((iy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE - 1)))
                as usize;
            let w4 = &wtab[alpha];

            // BORDER_CONSTANT: if any neighbor is outside, handle per-sample
            let dst_base = (oy * ow + ox) * ch;

            if (sx as u32) < (in_w - 1) as u32 && (sy as u32) < (in_h - 1) as u32 {
                // Fast path: all 4 neighbors inside
                let src_base = (sy * in_w + sx) as usize * ch;
                let src_row2 = src_base + in_w as usize * ch;
                for c in 0..ch {
                    let v = pixels[src_base + c] as i32 * w4[0]
                        + pixels[src_base + ch + c] as i32 * w4[1]
                        + pixels[src_row2 + c] as i32 * w4[2]
                        + pixels[src_row2 + ch + c] as i32 * w4[3]
                        + (1 << (INTER_REMAP_COEF_BITS - 1)); // +16384 rounding
                    out[dst_base + c] = (v >> INTER_REMAP_COEF_BITS).clamp(0, 255) as u8;
                }
            } else if sx >= in_w || sx + 1 < 0 || sy >= in_h || sy + 1 < 0 {
                // Fully outside → border constant (0), already zeroed
            } else {
                // Partially outside: fetch each neighbor individually
                for c in 0..ch {
                    let fetch = |x: i32, y: i32| -> i32 {
                        if x >= 0 && x < in_w && y >= 0 && y < in_h {
                            pixels[(y * in_w + x) as usize * ch + c] as i32
                        } else {
                            0 // border constant
                        }
                    };
                    let v = fetch(sx, sy) * w4[0]
                        + fetch(sx + 1, sy) * w4[1]
                        + fetch(sx, sy + 1) * w4[2]
                        + fetch(sx + 1, sy + 1) * w4[3]
                        + (1 << (INTER_REMAP_COEF_BITS - 1));
                    out[dst_base + c] = (v >> INTER_REMAP_COEF_BITS).clamp(0, 255) as u8;
                }
            }
        }
    }

    Ok(out)
}
