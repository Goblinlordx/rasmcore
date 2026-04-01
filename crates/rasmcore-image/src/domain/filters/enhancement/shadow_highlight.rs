//! Filter: shadow_highlight (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Shadow/highlight adjustment: independently lighten shadows and darken highlights.
///
/// Exact port of GEGL `gegl:shadows-highlights-correction` (darktable algorithm
/// by Ulrich Pegelow, GEGL port by Thomas Manni). Operates in CIE LAB.
///
/// Algorithm: soft-light blend in L* channel with compress-gated weight masks
/// and iterative application for strong settings. Adjusts a*/b* saturation
/// via shadows_ccorrect / highlights_ccorrect.
///
/// Reference: GEGL gegl:shadows-highlights (GPL3+).
/// Validated against GEGL (EXACT tier target).
#[rasmcore_macros::register_filter(name = "shadow_highlight", category = "enhancement")]
pub fn shadow_highlight(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ShadowHighlightParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    let shadows = config.shadows;
    let highlights = config.highlights;
    let whitepoint = config.whitepoint;
    let radius = config.radius;
    let compress = config.compress;
    let shadows_ccorrect = config.shadows_ccorrect;
    let highlights_ccorrect = config.highlights_ccorrect;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            shadow_highlight(r, &mut u, i8, config)
        });
    }

    let ch = channels(info.format);
    if ch < 3 {
        return Err(ImageError::UnsupportedFormat(
            "shadow_highlight requires RGB8 or RGBA8".into(),
        ));
    }

    let n = (info.width as usize) * (info.height as usize);

    // Identity fast path (matches GEGL's is_operation_a_nop)
    if shadows.abs() < 1e-6 && highlights.abs() < 1e-6 && whitepoint.abs() < 1e-6 {
        return Ok(pixels.to_vec());
    }

    // 1. Convert to CIE LAB
    let rgb_only: Vec<u8> = if ch == 4 {
        pixels
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect()
    } else {
        pixels.to_vec()
    };
    let rgb_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Rgb8,
        color_space: info.color_space,
    };
    let lab = crate::domain::color_spaces::image_rgb_to_lab(&rgb_only, &rgb_info)?;

    // 2. Compute blurred luminance in float precision.
    //    GEGL blurs in "Y float" (CIE linear luminance), NOT L* (perceptual).
    //    We compute Y from sRGB→linear→BT.709 weights, blur in float, then
    //    convert the blurred Y back to L* for the correction step.
    let w = info.width as usize;
    let h = info.height as usize;

    // Compute linear Y from sRGB input (linearize gamma, then BT.709 weights)
    let y_linear: Vec<f32> = (0..n)
        .map(|i| {
            let r_lin = srgb_to_linear(rgb_only[i * 3] as f32 / 255.0);
            let g_lin = srgb_to_linear(rgb_only[i * 3 + 1] as f32 / 255.0);
            let b_lin = srgb_to_linear(rgb_only[i * 3 + 2] as f32 / 255.0);
            0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
        })
        .collect();

    // Blur Y in float precision (matches GEGL's gaussian-blur on YaA float)
    let blurred_y = blur_1ch_f32(&y_linear, w, h, radius);

    // Convert blurred Y back to L* scale for the correction formula
    // L* = 116 * f(Y/Yn) - 16, where f(t) = t^(1/3) if t > (6/29)^3, else ...
    let blurred_l: Vec<f32> = blurred_y
        .iter()
        .map(|&y| {
            let t = y.max(0.0); // Y/Yn where Yn=1.0 (D65 normalized)
            let ft = if t > 0.008856 {
                t.cbrt()
            } else {
                7.787 * t + 16.0 / 116.0
            };
            (116.0 * ft - 16.0).max(0.0)
        })
        .collect();

    // 3. Pre-compute GEGL parameters (matches shadows-highlights-correction.c)
    let low_approximation: f32 = 0.01;
    let compress_f = (compress / 100.0).min(0.99);
    let whitepoint_f = 1.0 - whitepoint / 100.0;

    let shadows_100 = shadows / 100.0;
    let shadows_2 = 2.0 * shadows_100;
    let shadows_sign: f32 = if shadows_2 < 0.0 { -1.0 } else { 1.0 };

    let highlights_100 = highlights / 100.0;
    let highlights_2 = 2.0 * highlights_100;
    let highlights_sign_neg: f32 = if highlights_2 < 0.0 { 1.0 } else { -1.0 };

    let sc_100 = shadows_ccorrect / 100.0;
    let sc = (sc_100 - 0.5) * shadows_sign + 0.5;

    let hc_100 = highlights_ccorrect / 100.0;
    let hc = (hc_100 - 0.5) * highlights_sign_neg + 0.5;

    // 4. Process each pixel (exact GEGL algorithm)
    let mut lab_out = lab.clone();

    for i in 0..n {
        // GEGL normalizes: L/100, a/128, b/128
        let mut ta = [
            lab[i * 3] as f32 / 100.0,
            lab[i * 3 + 1] as f32 / 128.0,
            lab[i * 3 + 2] as f32 / 128.0,
        ];

        // tb0 = (100 - blurred_L) / 100 (inverted luminance)
        // tb0 = (100 - blurred_L) / 100 — inverted normalized luminance
        let mut tb0 = (100.0 - blurred_l[i]) / 100.0;

        // White point adjustment
        if ta[0] > 0.0 {
            ta[0] /= whitepoint_f;
        }
        if tb0 > 0.0 {
            tb0 /= whitepoint_f;
        }

        // --- Highlights processing ---
        if tb0 < 1.0 - compress_f {
            let mut h2 = highlights_2 * highlights_2;
            let hx = (1.0 - tb0 / (1.0 - compress_f)).min(1.0);

            while h2 > 0.0 {
                let la = ta[0];
                let la_inv = 1.0 - la;
                let lb =
                    (tb0 - 0.5) * highlights_sign_neg * if la_inv < 0.0 { -1.0 } else { 1.0 } + 0.5;

                let la_abs = la.abs();
                let lref = if la_abs > low_approximation {
                    1.0 / la_abs
                } else {
                    1.0 / low_approximation
                } * if la < 0.0 { -1.0 } else { 1.0 };

                let la_inv_abs = la_inv.abs();
                let href = if la_inv_abs > low_approximation {
                    1.0 / la_inv_abs
                } else {
                    1.0 / low_approximation
                } * if la_inv < 0.0 { -1.0 } else { 1.0 };

                let chunk = if h2 > 1.0 { 1.0 } else { h2 };
                let optrans = chunk * hx;
                h2 -= 1.0;

                // Soft-light blend
                let blended = if la > 0.5 {
                    1.0 - (1.0 - 2.0 * (la - 0.5)) * (1.0 - lb)
                } else {
                    2.0 * la * lb
                };
                ta[0] = la * (1.0 - optrans) + blended * optrans;

                // Color correction for a* and b*
                let cc_factor = ta[0] * lref * (1.0 - hc) + (1.0 - ta[0]) * href * hc;
                ta[1] = ta[1] * (1.0 - optrans) + ta[1] * cc_factor * optrans;
                ta[2] = ta[2] * (1.0 - optrans) + ta[2] * cc_factor * optrans;
            }
        }

        // --- Shadows processing ---
        if tb0 > compress_f {
            let mut s2 = shadows_2 * shadows_2;
            let sx = (tb0 / (1.0 - compress_f) - compress_f / (1.0 - compress_f)).min(1.0);

            while s2 > 0.0 {
                let la = ta[0];
                let la_inv = 1.0 - la;
                let lb = (tb0 - 0.5) * shadows_sign * if la_inv < 0.0 { -1.0 } else { 1.0 } + 0.5;

                let la_abs = la.abs();
                let lref = if la_abs > low_approximation {
                    1.0 / la_abs
                } else {
                    1.0 / low_approximation
                } * if la < 0.0 { -1.0 } else { 1.0 };

                let la_inv_abs = la_inv.abs();
                let href = if la_inv_abs > low_approximation {
                    1.0 / la_inv_abs
                } else {
                    1.0 / low_approximation
                } * if la_inv < 0.0 { -1.0 } else { 1.0 };

                let chunk = if s2 > 1.0 { 1.0 } else { s2 };
                let optrans = chunk * sx;
                s2 -= 1.0;

                let blended = if la > 0.5 {
                    1.0 - (1.0 - 2.0 * (la - 0.5)) * (1.0 - lb)
                } else {
                    2.0 * la * lb
                };
                ta[0] = la * (1.0 - optrans) + blended * optrans;

                let cc_factor = ta[0] * lref * sc + (1.0 - ta[0]) * href * (1.0 - sc);
                ta[1] = ta[1] * (1.0 - optrans) + ta[1] * cc_factor * optrans;
                ta[2] = ta[2] * (1.0 - optrans) + ta[2] * cc_factor * optrans;
            }
        }

        // De-normalize back to LAB
        lab_out[i * 3] = (ta[0] * 100.0) as f64;
        lab_out[i * 3 + 1] = (ta[1] * 128.0) as f64;
        lab_out[i * 3 + 2] = (ta[2] * 128.0) as f64;
    }

    // 5. Convert back to RGB
    let rgb_result = crate::domain::color_spaces::image_lab_to_rgb(&lab_out, &rgb_info)?;

    if ch == 4 {
        let mut result = vec![0u8; n * 4];
        for i in 0..n {
            result[i * 4] = rgb_result[i * 3];
            result[i * 4 + 1] = rgb_result[i * 3 + 1];
            result[i * 4 + 2] = rgb_result[i * 3 + 2];
            result[i * 4 + 3] = pixels[i * 4 + 3];
        }
        Ok(result)
    } else {
        Ok(rgb_result)
    }
}
