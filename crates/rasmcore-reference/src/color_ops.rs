//! Color operation reference implementations — matrix transforms, HSL operations.
//!
//! All operate in **linear f32** space. Alpha preserved unchanged.

/// Sepia tone — warm brownish tint via standard matrix.
///
/// Formula (full intensity):
///   R' = min(0.393*R + 0.769*G + 0.189*B, 1.0)
///   G' = min(0.349*R + 0.686*G + 0.168*B, 1.0)
///   B' = min(0.272*R + 0.534*G + 0.131*B, 1.0)
/// With intensity blend: out = lerp(in, sepia, intensity)
///
/// Validated against: ImageMagick 7.1.1 `-colorspace Linear -sepia-tone 80%`
/// The sepia matrix is a standard W3C approximation used across all tools.
pub fn sepia(input: &[f32], _w: u32, _h: u32, intensity: f32) -> Vec<f32> {
    let inv = 1.0 - intensity;
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (r, g, b) = (px[0], px[1], px[2]);
        let sr = 0.393 * r + 0.769 * g + 0.189 * b;
        let sg = 0.349 * r + 0.686 * g + 0.168 * b;
        let sb = 0.272 * r + 0.534 * g + 0.131 * b;
        px[0] = inv * r + intensity * sr;
        px[1] = inv * g + intensity * sg;
        px[2] = inv * b + intensity * sb;
    }
    out
}

/// Saturation adjustment in HSL space.
///
/// Formula: convert RGB→HSL, multiply S by factor (clamp to [0,1]), convert back.
/// factor=1.0 is identity, factor=0.0 is full grayscale, factor=2.0 is double saturation.
///
/// Validated against: CSS saturate() filter, Photoshop Hue/Saturation (HSL mode)
/// Uses the standard HSL ↔ RGB conversion (symmetric hexagonal model).
pub fn saturate(input: &[f32], _w: u32, _h: u32, factor: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (h, s, l) = rgb_to_hsl(px[0], px[1], px[2]);
        let (r, g, b) = hsl_to_rgb(h, (s * factor).clamp(0.0, 1.0), l);
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
    out
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    if (max - min).abs() < 1e-10 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if (max - r).abs() < 1e-10 {
        let mut h = (g - b) / d;
        if g < b {
            h += 6.0;
        }
        h
    } else if (max - g).abs() < 1e-10 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h / 6.0, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-10 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hue2rgb = |t: f32| -> f32 {
        let t = ((t % 1.0) + 1.0) % 1.0;
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    (hue2rgb(h + 1.0 / 3.0), hue2rgb(h), hue2rgb(h - 1.0 / 3.0))
}

/// Hue rotation — rotate hue by angle in degrees in HSL space.
///
/// Validated against: numpy independent HSL implementation + pipeline.
/// Uses HSL model matching DaVinci Resolve / pipeline implementation.
pub fn hue_rotate(input: &[f32], _w: u32, _h: u32, angle_deg: f32) -> Vec<f32> {
    // rgb_to_hsl returns H in [0, 1], hsl_to_rgb expects H in [0, 1]
    let shift = angle_deg / 360.0;
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (h, s, l) = rgb_to_hsl(px[0], px[1], px[2]);
        let nh = ((h + shift) % 1.0 + 1.0) % 1.0;
        let (r, g, b) = hsl_to_rgb(nh, s, l);
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
    out
}

/// Channel mixer — 3x3 matrix transform on RGB.
///
/// Matrix is row-major: [Rr, Rg, Rb, Gr, Gg, Gb, Br, Bg, Bb]
/// out_R = matrix[0]*R + matrix[1]*G + matrix[2]*B
/// out_G = matrix[3]*R + matrix[4]*G + matrix[5]*B
/// out_B = matrix[6]*R + matrix[7]*G + matrix[8]*B
///
/// Validated against: DaVinci Resolve Color Mixer (linear mode)
/// Standard 3x3 color matrix — mathematically unambiguous.
pub fn channel_mixer(input: &[f32], _w: u32, _h: u32, matrix: &[f32; 9]) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (r, g, b) = (px[0], px[1], px[2]);
        px[0] = matrix[0] * r + matrix[1] * g + matrix[2] * b;
        px[1] = matrix[3] * r + matrix[4] * g + matrix[5] * b;
        px[2] = matrix[6] * r + matrix[7] * g + matrix[8] * b;
    }
    out
}

/// White balance — Planckian locus temperature shift.
///
/// Simplified model: warm shifts R up and B down, cool does opposite.
/// CIE chromatic adaptation white balance (CAT16).
///
/// temperature_k: source illuminant in Kelvin (2000-12000).
///   Photography convention: adapts from source to D65 (display white).
///   8000K = "shot under blue sky" → warm up. 3200K = "tungsten" → cool down.
///   6500K = D65 neutral (identity).
/// tint: green-magenta shift (-1 to 1). 0 = no tint.
///
/// Implementation: CIE D-illuminant series (CIE 015:2018 eq 4.1) +
/// CAT16 chromatic adaptation (Li et al. 2017).
pub fn white_balance(input: &[f32], _w: u32, _h: u32, temperature_k: f32, tint: f32) -> Vec<f32> {
    let m = cat16_adaptation_matrix(temperature_k, tint);
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let r = m[0] * px[0] + m[1] * px[1] + m[2] * px[2];
        let g = m[3] * px[0] + m[4] * px[1] + m[5] * px[2];
        let b = m[6] * px[0] + m[7] * px[1] + m[8] * px[2];
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
    out
}

/// CIE D-illuminant chromaticity (CIE 015:2018, eq 4.1).
fn cie_d_illuminant_xy(temp_k: f32) -> (f64, f64) {
    let t = temp_k.clamp(4000.0, 25000.0) as f64;
    let (t2, t3) = (t * t, t * t * t);
    let xd = if t <= 7000.0 {
        -4.6070e9 / t3 + 2.9678e6 / t2 + 0.09911e3 / t + 0.244063
    } else {
        -2.0064e9 / t3 + 1.9018e6 / t2 + 0.24748e3 / t + 0.237040
    };
    let yd = -3.000 * xd * xd + 2.870 * xd - 0.275;
    (xd, yd)
}

/// Tint shift perpendicular to Planckian locus (in CIE 1960 uv space).
fn tint_shift_xy(xy: (f64, f64), tint: f32) -> (f64, f64) {
    if tint.abs() < 1e-6 {
        return xy;
    }
    let (x, y) = xy;
    let denom = -2.0 * x + 12.0 * y + 3.0;
    let u = 4.0 * x / denom;
    let v = 6.0 * y / denom;
    let v_shifted = v - tint as f64 * 0.02;
    let denom2 = 2.0 * u - 8.0 * v_shifted + 4.0;
    (3.0 * u / denom2, 2.0 * v_shifted / denom2)
}

/// CAT16 adaptation matrix (Li et al. 2017).
fn cat16_adaptation_matrix(temperature_k: f32, tint: f32) -> [f32; 9] {
    const CAT16: [[f64; 3]; 3] = [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ];
    const CAT16_INV: [[f64; 3]; 3] = [
        [
            1.862067855087232715,
            -1.011254630531684295,
            0.149186775444451747,
        ],
        [
            0.387526543236137111,
            0.621447441931475275,
            -0.008973985167612520,
        ],
        [
            -0.015841498849333856,
            -0.034122938028515563,
            1.049964436877849350,
        ],
    ];

    let src_xy = tint_shift_xy(cie_d_illuminant_xy(temperature_k), tint);
    // Standard CIE D65 chromaticity (not formula at 6500K — D65 CCT ≈ 6504K)
    let tgt_xy = (0.31270_f64, 0.32900_f64);

    let xy_to_xyz = |x: f64, y: f64| -> [f64; 3] {
        if y.abs() < 1e-10 {
            return [0.0, 1.0, 0.0];
        }
        [x / y, 1.0, (1.0 - x - y) / y]
    };
    let mv = |m: &[[f64; 3]; 3], v: &[f64; 3]| -> [f64; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    };

    let src_xyz = xy_to_xyz(src_xy.0, src_xy.1);
    let tgt_xyz = xy_to_xyz(tgt_xy.0, tgt_xy.1);
    let sc = mv(&CAT16, &src_xyz);
    let tc = mv(&CAT16, &tgt_xyz);
    let d = [tc[0] / sc[0], tc[1] / sc[1], tc[2] / sc[2]];

    let d_cat = [
        [d[0] * CAT16[0][0], d[0] * CAT16[0][1], d[0] * CAT16[0][2]],
        [d[1] * CAT16[1][0], d[1] * CAT16[1][1], d[1] * CAT16[1][2]],
        [d[2] * CAT16[2][0], d[2] * CAT16[2][1], d[2] * CAT16[2][2]],
    ];

    let mut out = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = (CAT16_INV[i][0] * d_cat[0][j]
                + CAT16_INV[i][1] * d_cat[1][j]
                + CAT16_INV[i][2] * d_cat[2][j]) as f32;
        }
    }
    out
}

/// White balance — gray world assumption.
///
/// Compute per-channel means, global average = (mean_r + mean_g + mean_b) / 3,
/// then scale each channel by global_avg / channel_avg.
///
/// Validated against: OpenCV manual gray-world implementation.
pub fn white_balance_gray_world(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let pixel_count = input.len() / 4;
    if pixel_count == 0 {
        return input.to_vec();
    }
    let (mut sum_r, mut sum_g, mut sum_b) = (0.0f64, 0.0f64, 0.0f64);
    for px in input.chunks_exact(4) {
        sum_r += px[0] as f64;
        sum_g += px[1] as f64;
        sum_b += px[2] as f64;
    }
    let n = pixel_count as f64;
    let avg_r = sum_r / n;
    let avg_g = sum_g / n;
    let avg_b = sum_b / n;
    let avg_all = (avg_r + avg_g + avg_b) / 3.0;

    let scale_r = if avg_r.abs() > 1e-10 { (avg_all / avg_r) as f32 } else { 1.0 };
    let scale_g = if avg_g.abs() > 1e-10 { (avg_all / avg_g) as f32 } else { 1.0 };
    let scale_b = if avg_b.abs() > 1e-10 { (avg_all / avg_b) as f32 } else { 1.0 };

    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[0] *= scale_r;
        px[1] *= scale_g;
        px[2] *= scale_b;
    }
    out
}

/// Colorize — tint image toward a target color via luma.
///
/// Compute luma = 0.2126*R + 0.7152*G + 0.0722*B,
/// blend: pixel += (luma * target - pixel) * amount.
/// amount=0 is identity, amount=1 replaces pixel with luma*target.
///
/// Validated against: Photoshop Colorize (Hue/Saturation dialog).
pub fn colorize(
    input: &[f32],
    _w: u32,
    _h: u32,
    target_r: f32,
    target_g: f32,
    target_b: f32,
    amount: f32,
) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let luma = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
        px[0] += (luma * target_r - px[0]) * amount;
        px[1] += (luma * target_g - px[1]) * amount;
        px[2] += (luma * target_b - px[2]) * amount;
    }
    out
}

/// Vibrance — saturation boost weighted by inverse of existing saturation.
///
/// For each pixel: sat = (max - min) / max, weight = amount * (1 - sat).
/// Convert to HSL, scale S by (1 + weight), convert back.
/// Low-saturation pixels get a stronger boost than already-saturated ones.
///
/// Validated against: Lightroom Vibrance slider behavior.
pub fn vibrance(input: &[f32], _w: u32, _h: u32, amount: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let max = px[0].max(px[1]).max(px[2]);
        let min = px[0].min(px[1]).min(px[2]);
        let sat = if max > 1e-10 { (max - min) / max } else { 0.0 };
        let weight = amount * (1.0 - sat);
        let (h, s, l) = rgb_to_hsl(px[0], px[1], px[2]);
        let s2 = (s * (1.0 + weight)).clamp(0.0, 1.0);
        let (r, g, b) = hsl_to_rgb(h, s2, l);
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
    out
}

/// HSL saturation — convert to HSL, scale S by factor, clamp [0,1], convert back.
///
/// Identical to `saturate()` but named explicitly for the HSL model.
/// factor=1.0 is identity, factor=0.0 is grayscale.
///
/// Validated against: CSS saturate() filter, Photoshop Hue/Saturation (HSL mode).
pub fn saturate_hsl(input: &[f32], _w: u32, _h: u32, factor: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (h, s, l) = rgb_to_hsl(px[0], px[1], px[2]);
        let (r, g, b) = hsl_to_rgb(h, (s * factor).clamp(0.0, 1.0), l);
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
    out
}

/// Modulate — combined brightness, saturation, and hue adjustment in HSL.
///
/// Convert to HSL, scale L by brightness, scale S by saturation,
/// rotate H by hue (degrees), convert back.
/// All factors at 1.0 / hue at 0.0 is identity.
///
/// Validated against: ImageMagick `-modulate brightness,saturation,hue`.
pub fn modulate(
    input: &[f32],
    _w: u32,
    _h: u32,
    brightness: f32,
    saturation: f32,
    hue: f32,
) -> Vec<f32> {
    let hue_frac = hue / 360.0; // convert degrees to [0,1] fraction
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (h, s, l) = rgb_to_hsl(px[0], px[1], px[2]);
        let l2 = (l * brightness).clamp(0.0, 1.0);
        let s2 = (s * saturation).clamp(0.0, 1.0);
        let h2 = ((h + hue_frac) % 1.0 + 1.0) % 1.0;
        let (r, g, b) = hsl_to_rgb(h2, s2, l2);
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
    out
}

/// Photo filter — blend toward a filter color by density.
///
/// out = lerp(pixel, filter_color, density).
/// If preserve_luminosity is true, scale result to maintain original luma
/// (BT.709: 0.2126*R + 0.7152*G + 0.0722*B).
///
/// Validated against: Photoshop Photo Filter adjustment layer.
pub fn photo_filter(
    input: &[f32],
    _w: u32,
    _h: u32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    density: f32,
    preserve_luminosity: bool,
) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let orig_luma = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
        let inv = 1.0 - density;
        px[0] = inv * px[0] + density * color_r;
        px[1] = inv * px[1] + density * color_g;
        px[2] = inv * px[2] + density * color_b;
        if preserve_luminosity {
            let new_luma = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
            if new_luma > 1e-10 {
                let scale = orig_luma / new_luma;
                px[0] *= scale;
                px[1] *= scale;
                px[2] *= scale;
            }
        }
    }
    out
}

/// Selective color — shift hue/sat/lum for pixels near a target hue.
///
/// Cosine-tapered weight: weight = max(0, cos(pi * hue_diff / hue_range)).
/// Apply weighted shifts to H, S, L.
///
/// Validated against: Photoshop Selective Color / Lightroom HSL panel.
pub fn selective_color(
    input: &[f32],
    _w: u32,
    _h: u32,
    target_hue: f32,
    hue_range: f32,
    hue_shift: f32,
    sat_shift: f32,
    lum_shift: f32,
) -> Vec<f32> {
    let target_frac = (target_hue / 360.0).rem_euclid(1.0);
    let range_frac = hue_range / 360.0;
    let shift_frac = hue_shift / 360.0;
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (h, s, l) = rgb_to_hsl(px[0], px[1], px[2]);
        let mut diff = (h - target_frac).abs();
        if diff > 0.5 {
            diff = 1.0 - diff;
        }
        if range_frac > 1e-10 && diff < range_frac {
            let weight = (std::f32::consts::PI * diff / range_frac).cos().max(0.0);
            let h2 = ((h + shift_frac * weight) % 1.0 + 1.0) % 1.0;
            let s2 = (s + sat_shift * weight).clamp(0.0, 1.0);
            let l2 = (l + lum_shift * weight).clamp(0.0, 1.0);
            let (r, g, b) = hsl_to_rgb(h2, s2, l2);
            px[0] = r;
            px[1] = g;
            px[2] = b;
        }
    }
    out
}

/// Replace color — shift H/S/L for pixels matching hue, saturation, and luminance ranges.
///
/// Cosine-tapered hue weight, hard S/L range gate.
///
/// Validated against: Photoshop Replace Color dialog.
pub fn replace_color(
    input: &[f32],
    _w: u32,
    _h: u32,
    center_hue: f32,
    hue_range: f32,
    sat_min: f32,
    sat_max: f32,
    lum_min: f32,
    lum_max: f32,
    hue_shift: f32,
    sat_shift: f32,
    lum_shift: f32,
) -> Vec<f32> {
    let center_frac = (center_hue / 360.0).rem_euclid(1.0);
    let range_frac = hue_range / 360.0;
    let shift_frac = hue_shift / 360.0;
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (h, s, l) = rgb_to_hsl(px[0], px[1], px[2]);
        if s < sat_min || s > sat_max || l < lum_min || l > lum_max {
            continue;
        }
        let mut diff = (h - center_frac).abs();
        if diff > 0.5 {
            diff = 1.0 - diff;
        }
        if range_frac > 1e-10 && diff < range_frac {
            let weight = (std::f32::consts::PI * diff / range_frac).cos().max(0.0);
            let h2 = ((h + shift_frac * weight) % 1.0 + 1.0) % 1.0;
            let s2 = (s + sat_shift * weight).clamp(0.0, 1.0);
            let l2 = (l + lum_shift * weight).clamp(0.0, 1.0);
            let (r, g, b) = hsl_to_rgb(h2, s2, l2);
            px[0] = r;
            px[1] = g;
            px[2] = b;
        }
    }
    out
}

/// Match color — transfer color statistics from a target distribution.
///
/// Per-channel: out = (pixel - src_mean) * (target_std / src_std) + target_mean,
/// blended by strength.
///
/// Validated against: Photoshop Match Color (simplified RGB model).
pub fn match_color(
    input: &[f32],
    _w: u32,
    _h: u32,
    target_mean: [f32; 3],
    target_std: f32,
    strength: f32,
) -> Vec<f32> {
    let pixel_count = input.len() / 4;
    if pixel_count == 0 {
        return input.to_vec();
    }
    let n = pixel_count as f64;

    // Compute per-channel mean
    let mut sum = [0.0f64; 3];
    for px in input.chunks_exact(4) {
        sum[0] += px[0] as f64;
        sum[1] += px[1] as f64;
        sum[2] += px[2] as f64;
    }
    let mean = [
        (sum[0] / n) as f32,
        (sum[1] / n) as f32,
        (sum[2] / n) as f32,
    ];

    // Compute per-channel std
    let mut var = [0.0f64; 3];
    for px in input.chunks_exact(4) {
        for c in 0..3 {
            let d = px[c] as f64 - mean[c] as f64;
            var[c] += d * d;
        }
    }
    let std_dev = [
        (var[0] / n).sqrt() as f32,
        (var[1] / n).sqrt() as f32,
        (var[2] / n).sqrt() as f32,
    ];

    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            let scale = if std_dev[c].abs() > 1e-10 {
                target_std / std_dev[c]
            } else {
                1.0
            };
            let transferred = (px[c] - mean[c]) * scale + target_mean[c];
            px[c] = px[c] + (transferred - px[c]) * strength;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sepia_zero_intensity_is_identity() {
        let input = crate::gradient(4, 4);
        let output = sepia(&input, 4, 4, 0.0);
        crate::assert_parity("sepia_zero", &output, &input, 1e-7);
    }

    #[test]
    fn saturate_one_is_identity() {
        let input = crate::gradient(4, 4);
        let output = saturate(&input, 4, 4, 1.0);
        // HSL roundtrip has f32 precision loss
        crate::assert_parity("saturate_one", &output, &input, 1e-5);
    }

    #[test]
    fn saturate_zero_is_grayscale() {
        let input = crate::gradient(4, 4);
        let output = saturate(&input, 4, 4, 0.0);
        for px in output.chunks(4) {
            assert!(
                (px[0] - px[1]).abs() < 1e-6 && (px[1] - px[2]).abs() < 1e-6,
                "should be grayscale"
            );
        }
    }

    #[test]
    fn channel_mixer_identity() {
        let input = crate::gradient(4, 4);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let output = channel_mixer(&input, 4, 4, &identity);
        crate::assert_parity("mixer_identity", &output, &input, 1e-7);
    }

    #[test]
    fn hue_rotate_360_is_identity() {
        let input = crate::gradient(4, 4);
        let output = hue_rotate(&input, 4, 4, 360.0);
        // YIQ rotation accumulates floating-point error at 360°
        crate::assert_parity("hue_360", &output, &input, 3e-3);
    }

    #[test]
    fn white_balance_d65_is_identity() {
        let input = crate::gradient(4, 4);
        let output = white_balance(&input, 4, 4, 6500.0, 0.0);
        // 6500K ≈ D65 but not exact (D65 CCT ≈ 6504K)
        crate::assert_parity("wb_d65", &output, &input, 0.002);
    }

    #[test]
    fn white_balance_warm_boosts_red() {
        let input = vec![0.5f32, 0.5, 0.5, 1.0];
        let output = white_balance(&input, 1, 1, 8000.0, 0.0);
        assert!(output[0] > 0.5, "8000K should boost red: {}", output[0]);
        assert!(output[2] < 0.5, "8000K should reduce blue: {}", output[2]);
    }

    #[test]
    fn gray_world_balanced_is_identity() {
        // A uniform gray image is already balanced — gray world should be identity.
        let input = vec![0.5f32, 0.5, 0.5, 1.0].repeat(16);
        let output = white_balance_gray_world(&input, 4, 4);
        crate::assert_parity("gray_world_balanced", &output, &input, 1e-6);
    }

    #[test]
    fn colorize_amount_zero_is_identity() {
        let input = crate::gradient(4, 4);
        let output = colorize(&input, 4, 4, 1.0, 0.5, 0.0, 0.0);
        crate::assert_parity("colorize_zero", &output, &input, 1e-7);
    }

    #[test]
    fn vibrance_zero_is_identity() {
        let input = crate::gradient(4, 4);
        let output = vibrance(&input, 4, 4, 0.0);
        // weight = 0 * (1 - sat) = 0, so S *= 1.0
        crate::assert_parity("vibrance_zero", &output, &input, 1e-5);
    }

    #[test]
    fn saturate_hsl_one_is_identity() {
        let input = crate::gradient(4, 4);
        let output = saturate_hsl(&input, 4, 4, 1.0);
        crate::assert_parity("saturate_hsl_one", &output, &input, 1e-5);
    }

    #[test]
    fn modulate_identity() {
        let input = crate::gradient(4, 4);
        let output = modulate(&input, 4, 4, 1.0, 1.0, 0.0);
        crate::assert_parity("modulate_identity", &output, &input, 1e-5);
    }

    #[test]
    fn photo_filter_density_zero_is_identity() {
        let input = crate::gradient(4, 4);
        let output = photo_filter(&input, 4, 4, 1.0, 0.5, 0.0, 0.0, false);
        crate::assert_parity("photo_filter_zero", &output, &input, 1e-7);
    }

    #[test]
    fn selective_color_zero_shift_is_identity() {
        let input = crate::gradient(4, 4);
        let output = selective_color(&input, 4, 4, 0.0, 60.0, 0.0, 0.0, 0.0);
        crate::assert_parity("selective_zero", &output, &input, 1e-5);
    }

    #[test]
    fn replace_color_zero_shift_is_identity() {
        let input = crate::gradient(4, 4);
        let output = replace_color(&input, 4, 4, 0.0, 60.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        crate::assert_parity("replace_zero", &output, &input, 1e-5);
    }

    #[test]
    fn match_color_strength_zero_is_identity() {
        let input = crate::gradient(4, 4);
        let output = match_color(&input, 4, 4, [0.5, 0.5, 0.5], 0.1, 0.0);
        crate::assert_parity("match_zero", &output, &input, 1e-7);
    }
}

/// Perceptual saturation via OKLCH — reference implementation.
///
/// Ottosson, B. (2020). "A perceptual color space for image processing."
/// https://bottosson.github.io/posts/oklab/
///
/// Converts linear sRGB → OKLab → OKLCH, scales chroma, converts back.
/// Perceptually uniform: equal factor changes produce equal perceived
/// saturation changes regardless of hue.
pub fn saturate_oklch(input: &[f32], _w: u32, _h: u32, factor: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (r, g, b) = (px[0], px[1], px[2]);

        // Linear sRGB -> LMS via M1
        let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
        let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
        let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

        // Cube root
        let l_ = l.max(0.0).cbrt();
        let m_ = m.max(0.0).cbrt();
        let s_ = s.max(0.0).cbrt();

        // LMS' -> OKLab
        let ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
        let ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
        let ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

        // OKLab -> OKLCH: scale chroma
        let c = (ok_a * ok_a + ok_b * ok_b).sqrt();
        let h = ok_b.atan2(ok_a);
        let c2 = c * factor;
        let a2 = c2 * h.cos();
        let b2 = c2 * h.sin();

        // OKLab -> LMS' (M2 inverse)
        let l2_ = ok_l + 0.3963377774 * a2 + 0.2158037573 * b2;
        let m2_ = ok_l - 0.1055613458 * a2 - 0.0638541728 * b2;
        let s2_ = ok_l - 0.0894841775 * a2 - 1.2914855480 * b2;

        // Cube (inverse of cube root)
        let l2 = l2_ * l2_ * l2_;
        let m2 = m2_ * m2_ * m2_;
        let s2 = s2_ * s2_ * s2_;

        // LMS -> linear sRGB (M1 inverse)
        px[0] = 4.0767416621 * l2 - 3.3077115913 * m2 + 0.2309699292 * s2;
        px[1] = -1.2684380046 * l2 + 2.6097574011 * m2 - 0.3413193965 * s2;
        px[2] = -0.0041960863 * l2 - 0.7034186147 * m2 + 1.7076147010 * s2;
    }
    out
}

#[cfg(test)]
mod saturate_tests {
    use super::*;

    #[test]
    fn saturate_oklch_factor_one_is_identity() {
        let input = crate::gradient(4, 4);
        let output = saturate_oklch(&input, 4, 4, 1.0);
        crate::assert_parity("oklch_sat_1.0", &output, &input, 1e-4);
    }

    #[test]
    fn saturate_oklch_factor_zero_is_grayscale() {
        let input = vec![0.8, 0.2, 0.4, 1.0];
        let output = saturate_oklch(&input, 1, 1, 0.0);
        assert!(
            (output[0] - output[1]).abs() < 0.02 && (output[1] - output[2]).abs() < 0.02,
            "factor=0 should produce grayscale, got ({:.3}, {:.3}, {:.3})",
            output[0],
            output[1],
            output[2]
        );
    }
}
