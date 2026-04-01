//! Pure color space conversion math — no image type dependencies. and color science operations.
//!
//! Color matrices use full 16-digit reference precision from colour-science.
#![allow(clippy::excessive_precision)]
//!
//! - CIE Lab (via XYZ, D65 illuminant)
//! - OKLab (Ottosson 2020)
//! - White balance (gray-world, manual temperature/tint)
//! - Bradford chromatic adaptation
//! - Perspective warp (4-point homography)



// ─── sRGB ↔ Linear ───────────────────────────────────────────────────────

#[inline]
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ─── ProPhoto RGB (ROMM RGB) ─────────────────────────────────────────────
//
// Primaries: R(0.7347, 0.2653) G(0.1596, 0.7468) B(0.0366, 0.0001)
// White point: D50
// Transfer function: gamma 1.8, linear below Et = 1/512 = 0.001953125
// Reference: ICC.1:2004-10, colour-science 0.4.7

const PROPHOTO_ET: f64 = 1.0 / 512.0; // 16 * (1/512)^1.8 = 1/512
const PROPHOTO_ET_LINEAR: f64 = 1.0 / 8192.0; // = Et / 16

#[inline]
fn prophoto_to_linear(v: f64) -> f64 {
    if v <= PROPHOTO_ET {
        v / 16.0
    } else {
        v.powf(1.8)
    }
}

#[inline]
fn linear_to_prophoto(v: f64) -> f64 {
    if v <= PROPHOTO_ET_LINEAR {
        v * 16.0
    } else {
        v.powf(1.0 / 1.8)
    }
}

/// ProPhoto RGB → XYZ (D50) matrix.
/// Derived from primaries R(0.7347,0.2653), G(0.1596,0.7468), B(0.0366,0.0001)
/// using our exact D50 white point. Row sums equal our Illuminant::D50 to f64 precision.
fn prophoto_rgb_to_xyz_d50(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = prophoto_to_linear(r);
    let gl = prophoto_to_linear(g);
    let bl = prophoto_to_linear(b);
    let x = 0.7830986009003157 * rl + 0.1532627928985585 * gl + 0.0279342826306939 * bl;
    let y = 0.2827767235862988 * rl + 0.7171469532371147 * gl + 0.0000763231765866 * bl;
    let z = 0.0000000000000000 * rl + 0.0898834424517862 * gl + 0.7352211600586739 * bl;
    (x, y, z)
}

/// XYZ (D50) → ProPhoto RGB. Exact inverse of the forward matrix.
fn xyz_d50_to_prophoto_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let rl = 1.3811931741006944 * x - 0.2886038185049336 * y - 0.0524476381088289 * z;
    let gl = -0.5446224939028348 * x + 1.5082327413132783 * y + 0.0205360323914797 * z;
    let bl = 0.0665820670677611 * x - 0.1843869003945875 * y + 1.3576243516096507 * z;
    (
        linear_to_prophoto(rl.clamp(0.0, 1.0)),
        linear_to_prophoto(gl.clamp(0.0, 1.0)),
        linear_to_prophoto(bl.clamp(0.0, 1.0)),
    )
}

/// Convert ProPhoto RGB to sRGB via XYZ (D50 → D65 Bradford adaptation).
pub fn prophoto_to_srgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x50, y50, z50) = prophoto_rgb_to_xyz_d50(r, g, b);
    let (x65, y65, z65) = bradford_adapt(x50, y50, z50, Illuminant::D50, Illuminant::D65);
    xyz_to_rgb(x65, y65, z65)
}

/// Convert sRGB to ProPhoto RGB via XYZ (D65 → D50 Bradford adaptation).
pub fn srgb_to_prophoto(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x65, y65, z65) = rgb_to_xyz(r, g, b);
    let (x50, y50, z50) = bradford_adapt(x65, y65, z65, Illuminant::D65, Illuminant::D50);
    xyz_d50_to_prophoto_rgb(x50, y50, z50)
}

// ─── Adobe RGB 1998 ──────────────────────────────────────────────────────
//
// Primaries: R(0.64, 0.33) G(0.21, 0.71) B(0.15, 0.06)
// White point: D65
// Transfer function: gamma 2.19921875 (= 563/256)
// Reference: Adobe RGB (1998) Color Image Encoding, colour-science 0.4.7

const ADOBE_GAMMA: f64 = 563.0 / 256.0; // 2.19921875

#[inline]
fn adobe_to_linear(v: f64) -> f64 {
    v.powf(ADOBE_GAMMA)
}

#[inline]
fn linear_to_adobe(v: f64) -> f64 {
    v.clamp(0.0, 1.0).powf(1.0 / ADOBE_GAMMA)
}

/// Adobe RGB → XYZ (D65) matrix.
/// Derived from primaries R(0.64,0.33), G(0.21,0.71), B(0.15,0.06)
/// using our exact D65 white point. Row sums equal D65 to f64 precision.
fn adobe_rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = adobe_to_linear(r);
    let gl = adobe_to_linear(g);
    let bl = adobe_to_linear(b);
    let x = 0.5766690429101305 * rl + 0.1855582379065463 * gl + 0.1882286462349947 * bl;
    let y = 0.2973449752505360 * rl + 0.6273635662554661 * gl + 0.0752914584939979 * bl;
    let z = 0.0270313613864123 * rl + 0.0706888525358272 * gl + 0.9913375368376386 * bl;
    (x, y, z)
}

/// XYZ (D65) → Adobe RGB. Exact inverse of forward matrix.
fn xyz_to_adobe_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let rl = 2.0415879038107470 * x - 0.5650069742788597 * y - 0.3447313507783297 * z;
    let gl = -0.9692436362808798 * x + 1.8759675015077206 * y + 0.0415550574071756 * z;
    let bl = 0.0134442806320312 * x - 0.1183623922310184 * y + 1.0151749943912058 * z;
    (
        linear_to_adobe(rl.clamp(0.0, 1.0)),
        linear_to_adobe(gl.clamp(0.0, 1.0)),
        linear_to_adobe(bl.clamp(0.0, 1.0)),
    )
}

/// Convert Adobe RGB to sRGB via XYZ (both D65, no adaptation needed).
pub fn adobe_to_srgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = adobe_rgb_to_xyz(r, g, b);
    xyz_to_rgb(x, y, z)
}

/// Convert sRGB to Adobe RGB via XYZ (both D65, no adaptation needed).
pub fn srgb_to_adobe(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    xyz_to_adobe_rgb(x, y, z)
}

// ─── CIE Lab (via XYZ, D65 illuminant) ───────────────────────────────────

/// D65 reference white point — derived from CIE chromaticity (0.3127, 0.329) with Y=1.
/// Matches colour-science's full-precision derivation.
const D65_X: f64 = 0.9504559270516716;
const D65_Y: f64 = 1.0;
const D65_Z: f64 = 1.0890577507598784;

/// sRGB to XYZ matrix (D65, IEC 61966-2-1).
/// Full 16-digit precision from colour-science 0.4.7:
///   colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ
fn rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);
    // Derived from sRGB primaries R(0.64,0.33), G(0.30,0.60), B(0.15,0.06)
    // using our exact D65 white point. Row sums equal D65 to f64 precision.
    let x = 0.4123907992659593 * rl + 0.3575843393838780 * gl + 0.1804807884018343 * bl;
    let y = 0.2126390058715102 * rl + 0.7151686787677560 * gl + 0.0721923153607337 * bl;
    let z = 0.0193308187155918 * rl + 0.1191947797946260 * gl + 0.9505321522496607 * bl;
    (x, y, z)
}

/// XYZ to sRGB. Exact inverse of forward matrix (cofactor/det, not from external source).
fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let rl = 3.2409699419045226 * x - 1.5373831775700939 * y - 0.4986107602930034 * z;
    let gl = -0.9692436362808796 * x + 1.8759675015077204 * y + 0.0415550574071756 * z;
    let bl = 0.0556300796969936 * x - 0.2039769588889765 * y + 1.0569715142428784 * z;
    (linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl))
}

#[inline]
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

#[inline]
fn lab_f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// Convert sRGB [0,1] to CIE Lab.
pub fn rgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    let fx = lab_f(x / D65_X);
    let fy = lab_f(y / D65_Y);
    let fz = lab_f(z / D65_Z);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);
    (l, a, b_val)
}

/// Convert CIE Lab to sRGB [0,1].
pub fn lab_to_rgb(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let x = D65_X * lab_f_inv(fx);
    let y = D65_Y * lab_f_inv(fy);
    let z = D65_Z * lab_f_inv(fz);
    xyz_to_rgb(x, y, z)
}
// ─── OKLab (Ottosson 2020) ───────────────────────────────────────────────
//
// Two paths provided per the Ottosson paper:
// 1. XYZ-based (default): sRGB → linear → XYZ → LMS → cbrt → OKLab
//    Matches colour-science 0.4.7 exactly.
// 2. Direct sRGB: sRGB → linear → LMS (direct matrix) → cbrt → OKLab
//    Uses the Ottosson paper's direct matrices, ~1e-4 difference from XYZ path.

/// Convert sRGB [0,1] to OKLab via XYZ intermediate.
/// Default path — matches colour-science reference exactly.
pub fn rgb_to_oklab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // rgb_to_xyz handles sRGB→linear internally
    let (x, y, z) = rgb_to_xyz(r, g, b);
    xyz_to_oklab(x, y, z)
}

/// Convert CIE XYZ to OKLab.
/// Uses colour-science's XYZ→LMS matrix (from Ottosson 2020).
pub fn xyz_to_oklab(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // M1: XYZ → LMS (colour-science MATRIX_1_XYZ_TO_LMS)
    let l = 0.8189330101 * x + 0.3618667424 * y - 0.1288597137 * z;
    let m = 0.0329845436 * x + 0.9293118715 * y + 0.0361456387 * z;
    let s = 0.0482003018 * x + 0.2643662691 * y + 0.6338517070 * z;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    // M2: LMS_cbrt → Lab (colour-science MATRIX_2_LMS_TO_LAB)
    let ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    (ok_l, ok_a, ok_b)
}

/// Convert OKLab to sRGB [0,1] via XYZ intermediate.
pub fn oklab_to_rgb(ok_l: f64, ok_a: f64, ok_b: f64) -> (f64, f64, f64) {
    let (x, y, z) = oklab_to_xyz(ok_l, ok_a, ok_b);
    let (rl, gl, bl) = xyz_to_rgb(x, y, z);
    (rl, gl, bl) // xyz_to_rgb already applies linear_to_srgb
}

/// Convert OKLab to CIE XYZ.
pub fn oklab_to_xyz(ok_l: f64, ok_a: f64, ok_b: f64) -> (f64, f64, f64) {
    // M2_inv: Lab → LMS_cbrt
    let l_ = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b;
    let m_ = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b;
    let s_ = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // M1_inv: LMS → XYZ (colour-science MATRIX_1_LMS_TO_XYZ)
    let x = 1.2270138511 * l - 0.5577999807 * m + 0.2812561490 * s;
    let y = -0.0405801784 * l + 1.1122568696 * m - 0.0716766787 * s;
    let z = -0.0763812845 * l - 0.4214819784 * m + 1.5861632204 * s;
    (x, y, z)
}

/// Convert sRGB [0,1] to OKLab via direct sRGB→LMS matrix (Ottosson paper path).
/// This is an alternative path — ~1e-4 difference from the XYZ-based default.
pub fn rgb_to_oklab_direct(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);

    // Direct linear_sRGB → LMS (Ottosson 2020 paper matrices)
    let l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl;
    let m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl;
    let s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    let ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    (ok_l, ok_a, ok_b)
}

/// Convert OKLab to sRGB [0,1] via direct LMS→sRGB matrix (Ottosson paper path).
pub fn oklab_to_rgb_direct(ok_l: f64, ok_a: f64, ok_b: f64) -> (f64, f64, f64) {
    let l_ = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b;
    let m_ = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b;
    let s_ = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // Direct LMS → linear_sRGB (Ottosson paper inverse)
    let rl = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let gl = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    (linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl))
}


// ─── Delta E (Color Difference Metrics) ──────────────────────────────────

/// Delta E 76 (CIE 1976) — Euclidean distance in Lab space.
pub fn delta_e_76(lab1: (f64, f64, f64), lab2: (f64, f64, f64)) -> f64 {
    let dl = lab1.0 - lab2.0;
    let da = lab1.1 - lab2.1;
    let db = lab1.2 - lab2.2;
    (dl * dl + da * da + db * db).sqrt()
}

/// Delta E 94 (CIE 1994) — weighted distance with lightness/chroma/hue terms.
/// `textile`: if true, uses textile weights (kL=2); false uses graphic arts (kL=1).
pub fn delta_e_94(lab1: (f64, f64, f64), lab2: (f64, f64, f64), textile: bool) -> f64 {
    let (k_l, k1, k2) = if textile {
        (2.0, 0.048, 0.014)
    } else {
        (1.0, 0.045, 0.015)
    };
    let dl = lab1.0 - lab2.0;
    let c1 = (lab1.1 * lab1.1 + lab1.2 * lab1.2).sqrt();
    let c2 = (lab2.1 * lab2.1 + lab2.2 * lab2.2).sqrt();
    let dc = c1 - c2;
    let da = lab1.1 - lab2.1;
    let db = lab1.2 - lab2.2;
    let dh_sq = da * da + db * db - dc * dc;
    let dh_sq = dh_sq.max(0.0); // numerical safety

    let sl = 1.0;
    let sc = 1.0 + k1 * c1;
    let sh = 1.0 + k2 * c1;

    let t1 = dl / (k_l * sl);
    let t2 = dc / sc;
    let t3_sq = dh_sq / (sh * sh);

    (t1 * t1 + t2 * t2 + t3_sq).sqrt()
}

/// Delta E 2000 (CIEDE2000) — most perceptually uniform color difference metric.
pub fn delta_e_2000(lab1: (f64, f64, f64), lab2: (f64, f64, f64)) -> f64 {
    use std::f64::consts::PI;
    let (l1, a1, b1) = lab1;
    let (l2, a2, b2) = lab2;

    let l_bar = (l1 + l2) / 2.0;
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) / 2.0;

    let c_bar_7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar_7 / (c_bar_7 + 25.0_f64.powi(7))).sqrt());

    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let c_bar_p = (c1p + c2p) / 2.0;

    let h1p = b1.atan2(a1p).to_degrees().rem_euclid(360.0);
    let h2p = b2.atan2(a2p).to_degrees().rem_euclid(360.0);

    let dh_abs = (h1p - h2p).abs();
    let dh_p = if c1p * c2p == 0.0 {
        0.0
    } else if dh_abs <= 180.0 {
        h2p - h1p
    } else if h2p <= h1p {
        h2p - h1p + 360.0
    } else {
        h2p - h1p - 360.0
    };

    let dl_p = l2 - l1;
    let dc_p = c2p - c1p;
    let dh_p_val = 2.0 * (c1p * c2p).sqrt() * (dh_p / 2.0 * PI / 180.0).sin();

    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if dh_abs <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_bar_p - 30.0) * PI / 180.0).cos()
        + 0.24 * ((2.0 * h_bar_p) * PI / 180.0).cos()
        + 0.32 * ((3.0 * h_bar_p + 6.0) * PI / 180.0).cos()
        - 0.20 * ((4.0 * h_bar_p - 63.0) * PI / 180.0).cos();

    let l_bar_50_sq = (l_bar - 50.0) * (l_bar - 50.0);
    let sl = 1.0 + 0.015 * l_bar_50_sq / (20.0 + l_bar_50_sq).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;

    let c_bar_p_7 = c_bar_p.powi(7);
    let rc = 2.0 * (c_bar_p_7 / (c_bar_p_7 + 25.0_f64.powi(7))).sqrt();
    let rt = -rc * (60.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp() * PI / 180.0).sin();

    let t1 = dl_p / sl;
    let t2 = dc_p / sc;
    let t3 = dh_p_val / sh;

    (t1 * t1 + t2 * t2 + t3 * t3 + rt * t2 * t3).sqrt()
}

// ─── CIE LCH (Cylindrical Lab) ──────────────────────────────────────────

/// Convert CIE Lab to LCH (Lightness, Chroma, Hue in degrees).
pub fn lab_to_lch(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let c = (a * a + b * b).sqrt();
    let h = b.atan2(a).to_degrees().rem_euclid(360.0);
    (l, c, h)
}

/// Convert CIE LCH to Lab.
pub fn lch_to_lab(l: f64, c: f64, h_deg: f64) -> (f64, f64, f64) {
    let h_rad = h_deg * std::f64::consts::PI / 180.0;
    let a = c * h_rad.cos();
    let b = c * h_rad.sin();
    (l, a, b)
}

// ─── CIE Luv ────────────────────────────────────────────────────────────

/// Convert CIE XYZ to CIE Luv (D65 illuminant).
pub fn xyz_to_luv(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let (xn, yn, zn) = (D65_X, D65_Y, D65_Z);
    let yr = y / yn;

    let l = if yr > (6.0 / 29.0_f64).powi(3) {
        116.0 * yr.cbrt() - 16.0
    } else {
        (29.0 / 3.0_f64).powi(3) * yr
    };

    let denom = x + 15.0 * y + 3.0 * z;
    let denom_n = xn + 15.0 * yn + 3.0 * zn;

    if denom == 0.0 {
        return (l, 0.0, 0.0);
    }

    let u_prime = 4.0 * x / denom;
    let v_prime = 9.0 * y / denom;
    let u_prime_n = 4.0 * xn / denom_n;
    let v_prime_n = 9.0 * yn / denom_n;

    let u = 13.0 * l * (u_prime - u_prime_n);
    let v = 13.0 * l * (v_prime - v_prime_n);
    (l, u, v)
}

/// Convert sRGB [0,1] to CIE Luv.
pub fn rgb_to_luv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    xyz_to_luv(x, y, z)
}

/// Convert CIE Luv to CIE XYZ (D65 illuminant).
pub fn luv_to_xyz(l: f64, u: f64, v: f64) -> (f64, f64, f64) {
    if l == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let (xn, yn, zn) = (D65_X, D65_Y, D65_Z);
    let denom_n = xn + 15.0 * yn + 3.0 * zn;
    let u_prime_n = 4.0 * xn / denom_n;
    let v_prime_n = 9.0 * yn / denom_n;

    let u_prime = u / (13.0 * l) + u_prime_n;
    let v_prime = v / (13.0 * l) + v_prime_n;

    let y = if l > 8.0 {
        yn * ((l + 16.0) / 116.0).powi(3)
    } else {
        yn * l * (3.0 / 29.0_f64).powi(3)
    };

    let x = y * 9.0 * u_prime / (4.0 * v_prime);
    let z = y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime);
    (x, y, z)
}

// ─── Bradford Chromatic Adaptation ───────────────────────────────────────

/// Standard illuminant white points (CIE XYZ, Y=1).
/// Derived from CIE chromaticity coordinates at full f64 precision,
/// matching colour-science's derivation via xy_to_XYZ().
#[derive(Debug, Clone, Copy)]
pub enum Illuminant {
    D50,
    D65,
    A,
}

impl Illuminant {
    pub fn xyz(self) -> (f64, f64, f64) {
        match self {
            // CIE xy (0.3457, 0.3585) → XYZ
            Illuminant::D50 => (0.964_295_676_429_568, 1.0, 0.8251046025104602),
            // CIE xy (0.3127, 0.329) → XYZ
            Illuminant::D65 => (D65_X, D65_Y, D65_Z),
            // CIE xy (0.44758, 0.40745) → XYZ
            Illuminant::A => (1.0984906123450726, 1.0, 0.355_798_257_454_903),
        }
    }
}

/// Bradford chromatic adaptation matrix: transform XYZ from one illuminant to another.
///
/// Full 16-digit precision from colour-science 0.4.7:
///   colour.adaptation.CHROMATIC_ADAPTATION_TRANSFORMS['Bradford']
pub fn bradford_adapt(x: f64, y: f64, z: f64, from: Illuminant, to: Illuminant) -> (f64, f64, f64) {
    // Bradford cone response matrix (colour-science 0.4.7)
    const M: [[f64; 3]; 3] = [
        [0.8951000, 0.2664000, -0.1614000],
        [-0.7502000, 1.7135000, 0.0367000],
        [0.0389000, -0.0685000, 1.0296000],
    ];
    // Inverse — numpy.linalg.inv(M) at full f64 precision
    const M_INV: [[f64; 3]; 3] = [
        [0.9869929054667123, -0.1470542564209900, 0.1599626516637315],
        [0.4323052697233945, 0.5183602715367776, 0.0492912282128556],
        [-0.0085286645751773, 0.0400428216540852, 0.9684866957875502],
    ];

    let (sx, sy, sz) = from.xyz();
    let (dx, dy, dz) = to.xyz();

    // Source cone response
    let s_rho = M[0][0] * sx + M[0][1] * sy + M[0][2] * sz;
    let s_gamma = M[1][0] * sx + M[1][1] * sy + M[1][2] * sz;
    let s_beta = M[2][0] * sx + M[2][1] * sy + M[2][2] * sz;

    // Destination cone response
    let d_rho = M[0][0] * dx + M[0][1] * dy + M[0][2] * dz;
    let d_gamma = M[1][0] * dx + M[1][1] * dy + M[1][2] * dz;
    let d_beta = M[2][0] * dx + M[2][1] * dy + M[2][2] * dz;

    // Transform input to cone space
    let rho = M[0][0] * x + M[0][1] * y + M[0][2] * z;
    let gamma = M[1][0] * x + M[1][1] * y + M[1][2] * z;
    let beta = M[2][0] * x + M[2][1] * y + M[2][2] * z;

    // Scale
    let rho_a = rho * (d_rho / s_rho);
    let gamma_a = gamma * (d_gamma / s_gamma);
    let beta_a = beta * (d_beta / s_beta);

    // Transform back to XYZ
    let xo = M_INV[0][0] * rho_a + M_INV[0][1] * gamma_a + M_INV[0][2] * beta_a;
    let yo = M_INV[1][0] * rho_a + M_INV[1][1] * gamma_a + M_INV[1][2] * beta_a;
    let zo = M_INV[2][0] * rho_a + M_INV[2][1] * gamma_a + M_INV[2][2] * beta_a;
    (xo, yo, zo)
}
