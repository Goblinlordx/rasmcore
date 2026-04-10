//! ACES 2.0 tonescale — parametric tone mapping curve.
//! Ported from OpenColorIO Transform.cpp (BSD-3-Clause).

use super::cam16::{JMhParams, j_to_y};
use super::constants::*;

/// Tonescale parameters (computed once per peak luminance).
#[derive(Debug, Clone)]
pub struct ToneScaleParams {
    pub n: f32,   // peak luminance in nits
    pub n_r: f32, // 100.0 (normalized white)
    pub g: f32,   // 1.15 (surround/contrast)
    pub t_1: f32, // 0.04 (shadow toe)
    pub c_t: f32, // mid-grey output luminance
    pub s_2: f32, // w_2 * m_1 * 100
    pub u_2: f32,
    pub m_2: f32,
    pub forward_limit: f32, // 8 * r_hit
    pub inverse_limit: f32, // n / (u_2 * n_r)
    pub log_peak: f32,      // log10(n / n_r)
}

/// Initialize tonescale params from peak luminance (matches OCIO exactly).
pub fn init_tonescale_params(peak_luminance: f32) -> ToneScaleParams {
    let n = peak_luminance;
    let n_r = 100.0_f32;
    let g = 1.15_f32;
    let c = 0.18_f32;
    let c_d = 10.013_f32;
    let w_g = 0.14_f32;
    let t_1 = 0.04_f32;
    let r_hit_min = 128.0_f32;
    let r_hit_max = 896.0_f32;

    let r_hit = r_hit_min + (r_hit_max - r_hit_min) * ((n / n_r).ln() / (10000.0_f32 / 100.0).ln());
    let m_0 = n / n_r;
    let m_1 = 0.5 * (m_0 + (m_0 * (m_0 + 4.0 * t_1)).sqrt());
    let u = ((r_hit / m_1) / ((r_hit / m_1) + 1.0)).powf(g);
    let _m = m_1 / u;
    let w_i = (n / 100.0_f32).log2();
    let c_t = c_d / n_r * (1.0 + w_i * w_g);
    let g_ip = 0.5 * (c_t + (c_t * (c_t + 4.0 * t_1)).sqrt());
    let g_ipp2 = -(m_1 * (g_ip / _m).powf(1.0 / g)) / ((g_ip / _m).powf(1.0 / g) - 1.0);
    let w_2 = c / g_ipp2;
    let s_2 = w_2 * m_1 * REFERENCE_LUMINANCE;
    let u_2 = ((r_hit / m_1) / ((r_hit / m_1) + w_2)).powf(g);
    let m_2 = m_1 / u_2;
    let inverse_limit = n / (u_2 * n_r);
    let forward_limit = 8.0 * r_hit;
    let log_peak = (n / n_r).log10();

    ToneScaleParams {
        n,
        n_r,
        g,
        t_1,
        c_t,
        s_2,
        u_2,
        m_2,
        forward_limit,
        inverse_limit,
        log_peak,
    }
}

/// Forward tonescale: scene luminance Y → display luminance Y_ts.
pub fn aces_tonescale_fwd(y_in: f32, pt: &ToneScaleParams) -> f32 {
    let f = pt.m_2 * (y_in / (y_in + pt.s_2)).powf(pt.g);
    (f * f / (f + pt.t_1)).max(0.0) * pt.n_r
}

/// Inverse tonescale: display luminance Y_ts → scene luminance Y.
pub fn aces_tonescale_inv(y_in: f32, pt: &ToneScaleParams) -> f32 {
    let y_ts_norm = y_in / REFERENCE_LUMINANCE;
    let z = y_ts_norm.max(0.0).min(pt.inverse_limit);
    let f = (z + (z * (4.0 * pt.t_1 + z)).sqrt()) / 2.0;
    pt.s_2 / ((pt.m_2 / f).powf(1.0 / pt.g) - 1.0)
}

/// Full tonescale in J domain: J_in → J_ts (applying tone curve via Y domain).
pub fn tonescale_fwd(j: f32, p: &JMhParams, pt: &ToneScaleParams) -> f32 {
    let j_abs = j.abs();
    let y_in = j_to_y(j_abs, p);
    let y_out = aces_tonescale_fwd(y_in, pt);
    let j_out = super::cam16::y_to_j(y_out, p);
    j_out.copysign(j)
}

/// Tonescale applied from achromatic channel A directly to J.
pub fn tonescale_a_to_j_fwd(a: f32, p: &JMhParams, pt: &ToneScaleParams) -> f32 {
    // A → Y → tonescale → Y_ts → J_ts
    let ra = p.a_w_j * a;
    let y = super::cam16::post_adaptation_cone_response_inv_pub(ra) / p.f_l_n;
    let y_ts = aces_tonescale_fwd(y, pt);
    let j = super::cam16::y_to_j(y_ts, p);
    j.copysign(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tonescale_sdr_midgrey() {
        let pt = init_tonescale_params(100.0);
        // Input Y is in relative luminance where 1.0 = reference_luminance (100 nits).
        // 18% grey = 0.18 relative = 18 nits scene luminance.
        // The tonescale expects absolute scene luminance, but OCIO feeds it
        // values from _J_to_Y which divides by F_L_n.
        // For direct testing: 18 nits scene should map to ~10 nits display.
        let y_out = aces_tonescale_fwd(18.0, &pt);
        assert!(
            y_out > 8.0 && y_out < 15.0,
            "18 nits scene should map to ~10 nits display: got {y_out}"
        );
    }

    #[test]
    fn tonescale_hdr_has_more_headroom() {
        let sdr = init_tonescale_params(100.0);
        let hdr = init_tonescale_params(1000.0);
        // Same input should produce higher output in HDR (more headroom)
        let y_sdr = aces_tonescale_fwd(1.0, &sdr);
        let y_hdr = aces_tonescale_fwd(1.0, &hdr);
        assert!(
            y_hdr > y_sdr,
            "HDR should have more headroom: {y_hdr} vs {y_sdr}"
        );
    }

    #[test]
    fn tonescale_fwd_inv_roundtrip() {
        let pt = init_tonescale_params(100.0);
        for y in [0.001, 0.01, 0.18, 0.5, 1.0, 5.0] {
            let y_ts = aces_tonescale_fwd(y, &pt);
            let y_back = aces_tonescale_inv(y_ts, &pt);
            assert!(
                (y_back - y).abs() < 0.01,
                "roundtrip at Y={y}: got {y_back}"
            );
        }
    }

    #[test]
    fn tonescale_black_stays_black() {
        let pt = init_tonescale_params(100.0);
        let y_out = aces_tonescale_fwd(0.0, &pt);
        assert!(y_out.abs() < 1e-6, "black should stay black: {y_out}");
    }

    #[test]
    fn tonescale_highlights_compress() {
        let pt = init_tonescale_params(100.0);
        // Very bright values should be compressed below peak (100 nits)
        let y_out = aces_tonescale_fwd(100.0, &pt);
        assert!(y_out < 100.0, "highlights should be compressed: {y_out}");
        // ACES tonescale compresses aggressively — 100x scene ≈ 46 nits is correct
        assert!(y_out > 30.0, "but not crushed: {y_out}");
    }
}
