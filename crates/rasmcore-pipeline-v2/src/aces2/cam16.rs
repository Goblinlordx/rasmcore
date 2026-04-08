//! CAM16 Color Appearance Model — RGB ↔ JMh conversion.
//! Ported from OpenColorIO Transform.cpp (BSD-3-Clause).

use super::constants::*;

/// CAM16 parameters for a specific color space (computed once per config).
#[derive(Debug, Clone)]
pub struct JMhParams {
    pub matrix_rgb_to_cam16_c: M33,
    pub matrix_cam16_c_to_rgb: M33,
    pub matrix_cone_response_to_aab: M33,
    pub matrix_aab_to_cone_response: M33,
    pub f_l_n: f32,
    pub cz: f32,
    pub inv_cz: f32,
    pub a_w_j: f32,
    pub inv_a_w_j: f32,
}

/// Model gamma: surround[1] * (1.48 + sqrt(Y_b / reference_luminance)).
fn model_gamma() -> f32 {
    SURROUND[1] * (1.48 + (Y_B / REFERENCE_LUMINANCE).sqrt())
}

/// Forward cone response compression: pow(|v|, 0.42) with sign preservation.
#[inline(always)]
fn post_adaptation_cone_response_fwd(v: f32) -> f32 {
    let abs_v = v.abs();
    let f_l_y = abs_v.powf(0.42);
    let ra = f_l_y / (CAM_NL_OFFSET + f_l_y);
    ra.copysign(v)
}

/// Inverse cone response compression.
#[inline(always)]
fn post_adaptation_cone_response_inv(v: f32) -> f32 {
    let abs_v = v.abs();
    let ra_lim = abs_v.min(0.99);
    let f_l_y = (CAM_NL_OFFSET * ra_lim) / (1.0 - ra_lim);
    let rc = f_l_y.powf(1.0 / 0.42);
    rc.copysign(v)
}

/// Achromatic response to lightness J.
#[inline(always)]
fn achromatic_to_j(a: f32, cz: f32) -> f32 {
    J_SCALE * a.powf(cz)
}

/// Lightness J to achromatic response.
#[inline(always)]
fn j_to_achromatic(j: f32, inv_cz: f32) -> f32 {
    (j / J_SCALE).powf(inv_cz)
}

/// Initialize JMhParams for a set of primaries.
/// Matches OCIO's init_JMhParams exactly.
pub fn init_jmh_params(prims: &[(f32, f32); 4]) -> JMhParams {
    let base_cone_to_aab: M33 = [
        2.0, 1.0, 1.0 / 20.0,
        1.0, -12.0 / 11.0, 1.0 / 11.0,
        1.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0,
    ];

    let matrix_16 = xyz_to_rgb_f33(&CAM16_PRIMS);
    let rgb_to_xyz = rgb_to_xyz_f33(prims);
    let xyz_w = mult_f3_f33(&[REFERENCE_LUMINANCE; 3], &rgb_to_xyz);
    let y_w = xyz_w[1];
    let rgb_w = mult_f3_f33(&xyz_w, &matrix_16);

    // Viewing condition parameters
    let k = 1.0 / (5.0 * L_A + 1.0);
    let k4 = k * k * k * k;
    let f_l = 0.2 * k4 * (5.0 * L_A) + 0.1 * (1.0 - k4).powi(2) * (5.0 * L_A).powf(1.0 / 3.0);
    let f_l_n = f_l / REFERENCE_LUMINANCE;
    let cz = model_gamma();
    let inv_cz = 1.0 / cz;

    let d_rgb = [
        f_l_n * y_w / rgb_w[0],
        f_l_n * y_w / rgb_w[1],
        f_l_n * y_w / rgb_w[2],
    ];

    let rgb_wc = [d_rgb[0] * rgb_w[0], d_rgb[1] * rgb_w[1], d_rgb[2] * rgb_w[2]];
    let rgb_aw = [
        post_adaptation_cone_response_fwd(rgb_wc[0]),
        post_adaptation_cone_response_fwd(rgb_wc[1]),
        post_adaptation_cone_response_fwd(rgb_wc[2]),
    ];

    let cone_to_aab = mult_f33_f33(
        &scale_f33(&IDENTITY_M33, &[CAM_NL_SCALE; 3]),
        &base_cone_to_aab,
    );
    let a_w = cone_to_aab[0] * rgb_aw[0] + cone_to_aab[1] * rgb_aw[1] + cone_to_aab[2] * rgb_aw[2];
    let a_w_j = {
        let f = f_l.powf(0.42);
        f / (CAM_NL_OFFSET + f)
    };
    let inv_a_w_j = 1.0 / a_w_j;

    // Build adapted matrices
    let rgb_to_cam16 = mult_f33_f33(
        &rgb_to_rgb_f33(prims, &CAM16_PRIMS),
        &scale_f33(&IDENTITY_M33, &[REFERENCE_LUMINANCE; 3]),
    );
    let matrix_rgb_to_cam16_c = mult_f33_f33(
        &scale_f33(&IDENTITY_M33, &d_rgb),
        &rgb_to_cam16,
    );
    let matrix_cam16_c_to_rgb = invert_f33(&matrix_rgb_to_cam16_c);

    let matrix_cone_response_to_aab: M33 = [
        cone_to_aab[0] / a_w, cone_to_aab[1] / a_w, cone_to_aab[2] / a_w,
        cone_to_aab[3] * 43.0 * SURROUND[2], cone_to_aab[4] * 43.0 * SURROUND[2], cone_to_aab[5] * 43.0 * SURROUND[2],
        cone_to_aab[6] * 43.0 * SURROUND[2], cone_to_aab[7] * 43.0 * SURROUND[2], cone_to_aab[8] * 43.0 * SURROUND[2],
    ];
    let matrix_aab_to_cone_response = invert_f33(&matrix_cone_response_to_aab);

    JMhParams {
        matrix_rgb_to_cam16_c,
        matrix_cam16_c_to_rgb,
        matrix_cone_response_to_aab,
        matrix_aab_to_cone_response,
        f_l_n,
        cz,
        inv_cz,
        a_w_j,
        inv_a_w_j,
    }
}

/// RGB → Aab (adapted achromatic + color opponent signals).
pub fn rgb_to_aab(rgb: &F3, p: &JMhParams) -> F3 {
    let rgb_m = mult_f3_f33(rgb, &p.matrix_rgb_to_cam16_c);
    let rgb_a = [
        post_adaptation_cone_response_fwd(rgb_m[0]),
        post_adaptation_cone_response_fwd(rgb_m[1]),
        post_adaptation_cone_response_fwd(rgb_m[2]),
    ];
    mult_f3_f33(&rgb_a, &p.matrix_cone_response_to_aab)
}

/// Aab → JMh (lightness, colorfulness, hue).
pub fn aab_to_jmh(aab: &F3, p: &JMhParams) -> F3 {
    if aab[0] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let j = achromatic_to_j(aab[0], p.cz);
    let m = (aab[1] * aab[1] + aab[2] * aab[2]).sqrt();
    let h = from_radians_unwrapped(aab[2].atan2(aab[1]));
    [j, m, h]
}

/// RGB → JMh (convenience).
pub fn rgb_to_jmh(rgb: &F3, p: &JMhParams) -> F3 {
    aab_to_jmh(&rgb_to_aab(rgb, p), p)
}

/// JMh → Aab (using precomputed cos/sin of hue).
pub fn jmh_to_aab_with_trig(jmh: &F3, cos_hr: f32, sin_hr: f32, p: &JMhParams) -> F3 {
    let a = j_to_achromatic(jmh[0], p.inv_cz);
    let ab_a = jmh[1] * cos_hr;
    let ab_b = jmh[1] * sin_hr;
    [a, ab_a, ab_b]
}

/// JMh → Aab (computing trig from hue).
pub fn jmh_to_aab(jmh: &F3, p: &JMhParams) -> F3 {
    let h_rad = to_radians(jmh[2]);
    jmh_to_aab_with_trig(jmh, h_rad.cos(), h_rad.sin(), p)
}

/// Aab → RGB.
pub fn aab_to_rgb(aab: &F3, p: &JMhParams) -> F3 {
    let rgb_a = mult_f3_f33(aab, &p.matrix_aab_to_cone_response);
    let rgb_m = [
        post_adaptation_cone_response_inv(rgb_a[0]),
        post_adaptation_cone_response_inv(rgb_a[1]),
        post_adaptation_cone_response_inv(rgb_a[2]),
    ];
    mult_f3_f33(&rgb_m, &p.matrix_cam16_c_to_rgb)
}

/// JMh → RGB.
pub fn jmh_to_rgb(jmh: &F3, p: &JMhParams) -> F3 {
    aab_to_rgb(&jmh_to_aab(jmh, p), p)
}

/// J → Y (luminance) via achromatic channel.
pub fn j_to_y(j: f32, p: &JMhParams) -> f32 {
    let a = j_to_achromatic(j.abs(), p.inv_cz);
    let ra = p.a_w_j * a;
    post_adaptation_cone_response_inv(ra) / p.f_l_n
}

/// Y → J.
pub fn y_to_j(y: f32, p: &JMhParams) -> f32 {
    let ra = {
        let v = (y.abs() * p.f_l_n).powf(0.42);
        v / (CAM_NL_OFFSET + v)
    };
    let j = achromatic_to_j(ra * p.inv_a_w_j, p.cz);
    j.copysign(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cam16_roundtrip_ap0() {
        let p = init_jmh_params(&AP0_PRIMS);
        let rgb = [0.18, 0.18, 0.18]; // 18% grey
        let jmh = rgb_to_jmh(&rgb, &p);
        let back = jmh_to_rgb(&jmh, &p);
        for i in 0..3 {
            assert!((back[i] - rgb[i]).abs() < 0.001,
                "AP0 roundtrip ch{i}: {} vs {}", back[i], rgb[i]);
        }
    }

    #[test]
    fn cam16_roundtrip_ap1() {
        let p = init_jmh_params(&AP1_PRIMS);
        let rgb = [0.5, 0.3, 0.1]; // colored pixel
        let jmh = rgb_to_jmh(&rgb, &p);
        let back = jmh_to_rgb(&jmh, &p);
        for i in 0..3 {
            assert!((back[i] - rgb[i]).abs() < 0.001,
                "AP1 roundtrip ch{i}: {} vs {}", back[i], rgb[i]);
        }
    }

    #[test]
    fn grey_has_zero_chroma() {
        let p = init_jmh_params(&AP0_PRIMS);
        let jmh = rgb_to_jmh(&[0.18, 0.18, 0.18], &p);
        assert!(jmh[1] < 0.01, "18% grey should have near-zero chroma: M={}", jmh[1]);
    }

    #[test]
    fn black_maps_to_zero() {
        let p = init_jmh_params(&AP0_PRIMS);
        let jmh = rgb_to_jmh(&[0.0, 0.0, 0.0], &p);
        assert!(jmh[0].abs() < 1e-6, "black J={}", jmh[0]);
        assert!(jmh[1].abs() < 1e-6, "black M={}", jmh[1]);
    }

    #[test]
    fn y_to_j_roundtrip() {
        let p = init_jmh_params(&AP0_PRIMS);
        let y = 0.18;
        let j = y_to_j(y, &p);
        let y_back = j_to_y(j, &p);
        assert!((y_back - y).abs() < 0.001, "Y roundtrip: {} vs {}", y_back, y);
    }
}
