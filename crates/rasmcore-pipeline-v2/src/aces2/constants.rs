//! ACES 2.0 constants and math types — ported from OpenColorIO Common.h and MatrixLib.h.
//! SPDX-License-Identifier: BSD-3-Clause
//! Copyright Contributors to the OpenColorIO Project.

use std::f32::consts::PI;

pub const HUE_LIMIT: f32 = 360.0;

// ─── Table dimensions ──────────────────────────────────────────────────────
pub const TABLE_NOMINAL_SIZE: usize = 360;
pub const TABLE_TOTAL_SIZE: usize = TABLE_NOMINAL_SIZE + 3; // 1 lower wrap + 360 + 2 upper wrap
pub const TABLE_BASE_INDEX: usize = 1;
pub const TABLE_FIRST_NOMINAL: usize = TABLE_BASE_INDEX;
pub const TABLE_UPPER_WRAP: usize = TABLE_BASE_INDEX + TABLE_NOMINAL_SIZE;

// ─── CAM16 viewing conditions ──────────────────────────────────────────────
pub const REFERENCE_LUMINANCE: f32 = 100.0;
pub const L_A: f32 = 100.0;
pub const Y_B: f32 = 20.0;
pub const SURROUND: [f32; 3] = [0.9, 0.59, 0.9];
pub const J_SCALE: f32 = 100.0;
pub const CAM_NL_OFFSET: f32 = 27.13;  // 0.2713 * 100
pub const CAM_NL_SCALE: f32 = 400.0;   // 4.0 * 100

// ─── Chroma compression ────────────────────────────────────────────────────
pub const CHROMA_COMPRESS: f32 = 2.4;
pub const CHROMA_COMPRESS_FACT: f32 = 3.3;
pub const CHROMA_EXPAND: f32 = 1.3;
pub const CHROMA_EXPAND_FACT: f32 = 0.69;
pub const CHROMA_EXPAND_THR: f32 = 0.5;
pub const CHROMA_COMPRESS_WEIGHTS: [f32; 8] = [
    11.34072, 16.46899, 7.88380, 0.0,
    14.66441, -6.37224, 9.19364, 77.12896,
];

// ─── Gamut compression ─────────────────────────────────────────────────────
pub const SMOOTH_CUSPS: f32 = 0.12;
pub const SMOOTH_M: f32 = 0.27;
pub const CUSP_MID_BLEND: f32 = 1.3;
pub const FOCUS_GAIN_BLEND: f32 = 0.3;
pub const FOCUS_ADJUST_GAIN_INV: f32 = 1.0 / 0.55;
pub const FOCUS_DISTANCE: f32 = 1.35;
pub const FOCUS_DISTANCE_SCALING: f32 = 1.75;
pub const COMPRESSION_THRESHOLD: f32 = 0.75;

// ─── Primaries (CIE xy) ───────────────────────────────────────────────────
pub const CAM16_PRIMS: [(f32,f32); 4] = [(0.8336,0.1735), (2.3854,-1.4659), (0.087,-0.125), (0.333,0.333)];
pub const AP0_PRIMS: [(f32,f32); 4] = [(0.7347,0.2653), (0.0,1.0), (0.0001,-0.077), (0.32168,0.33767)];
pub const AP1_PRIMS: [(f32,f32); 4] = [(0.713,0.293), (0.165,0.83), (0.128,0.044), (0.32168,0.33767)];

// Limiting (display) gamut primaries
pub const REC709_PRIMS: [(f32,f32); 4] = [(0.64,0.33), (0.30,0.60), (0.15,0.06), (0.3127,0.3290)];
pub const P3_D65_PRIMS: [(f32,f32); 4] = [(0.680,0.320), (0.265,0.690), (0.150,0.060), (0.3127,0.3290)];
pub const REC2020_PRIMS: [(f32,f32); 4] = [(0.708,0.292), (0.170,0.797), (0.131,0.046), (0.3127,0.3290)];

// ─── Table generation ──────────────────────────────────────────────────────
pub const CUSP_CORNER_COUNT: usize = 6;
pub const TOTAL_CORNER_COUNT: usize = CUSP_CORNER_COUNT + 2;
pub const MAX_SORTED_CORNERS: usize = 2 * CUSP_CORNER_COUNT;
pub const REACH_CUSP_TOLERANCE: f32 = 1e-3;
pub const DISPLAY_CUSP_TOLERANCE: f32 = 1e-7;
pub const GAMMA_MINIMUM: f32 = 0.0;
pub const GAMMA_MAXIMUM: f32 = 5.0;
pub const GAMMA_SEARCH_STEP: f32 = 0.4;
pub const GAMMA_ACCURACY: f32 = 1e-5;

// ─── Math types ────────────────────────────────────────────────────────────
pub type F3 = [f32; 3];
pub type F2 = [f32; 2];
pub type M33 = [f32; 9];

pub const IDENTITY_M33: M33 = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0];

// ─── Matrix math ───────────────────────────────────────────────────────────

#[inline(always)]
pub fn mult_f3_f33(v: &F3, m: &M33) -> F3 {
    [v[0]*m[0]+v[1]*m[1]+v[2]*m[2], v[0]*m[3]+v[1]*m[4]+v[2]*m[5], v[0]*m[6]+v[1]*m[7]+v[2]*m[8]]
}

pub fn mult_f33_f33(a: &M33, b: &M33) -> M33 {
    [
        a[0]*b[0]+a[1]*b[3]+a[2]*b[6], a[0]*b[1]+a[1]*b[4]+a[2]*b[7], a[0]*b[2]+a[1]*b[5]+a[2]*b[8],
        a[3]*b[0]+a[4]*b[3]+a[5]*b[6], a[3]*b[1]+a[4]*b[4]+a[5]*b[7], a[3]*b[2]+a[4]*b[5]+a[5]*b[8],
        a[6]*b[0]+a[7]*b[3]+a[8]*b[6], a[6]*b[1]+a[7]*b[4]+a[8]*b[7], a[6]*b[2]+a[7]*b[5]+a[8]*b[8],
    ]
}

/// OCIO's scale_f33: scales column i of identity by scale[i], then multiplies.
pub fn scale_f33(m: &M33, scale: &F3) -> M33 {
    [m[0]*scale[0],m[3],m[6], m[1],m[4]*scale[1],m[7], m[2],m[5],m[8]*scale[2]]
}

pub fn mult_f_f3(s: f32, v: &F3) -> F3 { [s*v[0], s*v[1], s*v[2]] }

pub fn invert_f33(m: &M33) -> M33 {
    let det = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
    let d = 1.0 / det;
    [(m[4]*m[8]-m[5]*m[7])*d, (m[2]*m[7]-m[1]*m[8])*d, (m[1]*m[5]-m[2]*m[4])*d,
     (m[5]*m[6]-m[3]*m[8])*d, (m[0]*m[8]-m[2]*m[6])*d, (m[2]*m[3]-m[0]*m[5])*d,
     (m[3]*m[7]-m[4]*m[6])*d, (m[1]*m[6]-m[0]*m[7])*d, (m[0]*m[4]-m[1]*m[3])*d]
}

/// Build RGB→XYZ from primaries (equal-energy illuminant, no chromatic adaptation).
pub fn rgb_to_xyz_f33(p: &[(f32,f32); 4]) -> M33 {
    let (xr,yr) = p[0]; let zr = 1.0-xr-yr;
    let (xg,yg) = p[1]; let zg = 1.0-xg-yg;
    let (xb,yb) = p[2]; let zb = 1.0-xb-yb;
    let (xw,yw) = p[3]; let zw = 1.0-xw-yw;
    let wx = xw/yw; let wz = zw/yw;
    let raw = [xr,xg,xb, yr,yg,yb, zr,zg,zb];
    let inv = invert_f33(&raw);
    let s = mult_f3_f33(&[wx, 1.0, wz], &inv);
    [s[0]*xr,s[1]*xg,s[2]*xb, s[0]*yr,s[1]*yg,s[2]*yb, s[0]*zr,s[1]*zg,s[2]*zb]
}

pub fn xyz_to_rgb_f33(p: &[(f32,f32); 4]) -> M33 { invert_f33(&rgb_to_xyz_f33(p)) }

pub fn rgb_to_rgb_f33(src: &[(f32,f32); 4], dst: &[(f32,f32); 4]) -> M33 {
    mult_f33_f33(&xyz_to_rgb_f33(dst), &rgb_to_xyz_f33(src))
}

pub fn wrap_to_hue_limit(hue: f32) -> f32 {
    let mut y = hue % HUE_LIMIT;
    if y < 0.0 { y += HUE_LIMIT; }
    y
}
pub fn to_radians(deg: f32) -> f32 { PI * deg / 180.0 }
pub fn from_radians_unwrapped(rad: f32) -> f32 { let mut y = 180.0*rad/PI; if y<0.0{y+=HUE_LIMIT;} y }
pub fn lerpf(a: f32, b: f32, t: f32) -> f32 { (b-a)*t + a }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_invert_roundtrip() {
        let m: M33 = [2.0,1.0,0.0, 0.0,3.0,1.0, 1.0,0.0,2.0];
        let inv = invert_f33(&m);
        let r = mult_f33_f33(&m, &inv);
        for i in 0..3 { for j in 0..3 {
            let exp = if i==j {1.0} else {0.0};
            assert!((r[i*3+j]-exp).abs() < 1e-5, "({i},{j})");
        }}
    }

    #[test]
    fn ap0_to_xyz_produces_valid_matrix() {
        let m = rgb_to_xyz_f33(&AP0_PRIMS);
        // White point should map to equal XYZ
        let w = mult_f3_f33(&[1.0, 1.0, 1.0], &m);
        // Y should be close to 1.0 (normalized white)
        assert!((w[1] - 1.0).abs() < 0.1, "Y = {}", w[1]);
    }

    #[test]
    fn hue_wrapping() {
        assert!((wrap_to_hue_limit(0.0)).abs() < 1e-6);
        assert!((wrap_to_hue_limit(-10.0) - 350.0).abs() < 1e-4);
        assert!((wrap_to_hue_limit(370.0) - 10.0).abs() < 1e-4);
    }
}
