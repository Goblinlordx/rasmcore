//! ACES 2.0 chroma and gamut compression — ported from OpenColorIO (BSD-3-Clause).
//!
//! Chroma compression reduces colorfulness at luminance extremes.
//! Gamut compression maps out-of-gamut colors to the display boundary.

use super::constants::*;
use super::cam16::*;
use super::tonescale::*;

// ─── Chroma Compress Params ────────────────────────────────────────────────

/// Chroma compression parameters (computed once per peak luminance).
#[derive(Debug, Clone)]
pub struct ChromaCompressParams {
    pub sat: f32,
    pub sat_thr: f32,
    pub compr: f32,
    pub chroma_compress_scale: f32,
}

/// Initialize chroma compress params from peak luminance.
pub fn init_chroma_compress_params(peak_luminance: f32, ts: &ToneScaleParams) -> ChromaCompressParams {
    let compr = CHROMA_COMPRESS + CHROMA_COMPRESS * CHROMA_COMPRESS_FACT * ts.log_peak;
    let sat = (CHROMA_EXPAND - CHROMA_EXPAND * CHROMA_EXPAND_FACT * ts.log_peak).max(0.2);
    let sat_thr = CHROMA_EXPAND_THR / ts.n;
    let chroma_compress_scale = (0.03379 * peak_luminance).powf(0.30596) - 0.45135;

    ChromaCompressParams { sat, sat_thr, compr, chroma_compress_scale }
}

// ─── Shared Compression Params ─────────────────────────────────────────────

/// Shared parameters for chroma and gamut compression.
#[derive(Debug, Clone)]
pub struct SharedCompressionParams {
    pub limit_j_max: f32,
    pub model_gamma_inv: f32,
    pub reach_m_table: Vec<f32>, // TABLE_TOTAL_SIZE entries
}

/// Resolved per-hue compression parameters.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedCompressionParams {
    pub limit_j_max: f32,
    pub model_gamma_inv: f32,
    pub reach_max_m: f32,
}

fn model_gamma() -> f32 {
    SURROUND[1] * (1.48 + (Y_B / REFERENCE_LUMINANCE).sqrt())
}

/// Build the reach_m_table: for each hue degree, find max M where
/// JMh_to_RGB has no negative components at limit_j_max.
fn make_reach_m_table(reach_params: &JMhParams, limit_j_max: f32) -> Vec<f32> {
    let mut table = vec![0.0_f32; TABLE_TOTAL_SIZE];

    for i in 0..TABLE_NOMINAL_SIZE {
        let hue = i as f32;
        let h_rad = to_radians(hue);
        let cos_h = h_rad.cos();
        let sin_h = h_rad.sin();

        // Binary search for maximum M
        let mut lo = 0.0_f32;
        let mut hi = 200.0_f32; // generous upper bound
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let jmh = [limit_j_max, mid, hue];
            let aab = jmh_to_aab_with_trig(&jmh, cos_h, sin_h, reach_params);
            let rgb = aab_to_rgb(&aab, reach_params);
            if rgb[0] >= 0.0 && rgb[1] >= 0.0 && rgb[2] >= 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        table[TABLE_BASE_INDEX + i] = lo;
    }

    // Wrap entries
    table[0] = table[TABLE_BASE_INDEX + TABLE_NOMINAL_SIZE - 1]; // lower wrap
    table[TABLE_UPPER_WRAP] = table[TABLE_BASE_INDEX];            // upper wrap 1
    if TABLE_UPPER_WRAP + 1 < TABLE_TOTAL_SIZE {
        table[TABLE_UPPER_WRAP + 1] = table[TABLE_BASE_INDEX + 1]; // upper wrap 2
    }

    table
}

/// Initialize shared compression parameters.
pub fn init_shared_compression_params(
    peak_luminance: f32,
    input_params: &JMhParams,
    reach_params: &JMhParams,
) -> SharedCompressionParams {
    let limit_j_max = y_to_j(peak_luminance, input_params);
    let model_gamma_inv = 1.0 / model_gamma();
    let reach_m_table = make_reach_m_table(reach_params, limit_j_max);

    SharedCompressionParams { limit_j_max, model_gamma_inv, reach_m_table }
}

/// Resolve per-hue parameters from shared params.
pub fn resolve_compression_params(hue: f32, p: &SharedCompressionParams) -> ResolvedCompressionParams {
    let reach_max_m = reach_m_from_table(hue, &p.reach_m_table);
    ResolvedCompressionParams {
        limit_j_max: p.limit_j_max,
        model_gamma_inv: p.model_gamma_inv,
        reach_max_m,
    }
}

/// Interpolate reach_m from table (uniform 1-degree spacing).
fn reach_m_from_table(hue: f32, table: &[f32]) -> f32 {
    let base = hue as usize;
    let t = hue - base as f32;
    let i_lo = base + TABLE_FIRST_NOMINAL;
    let i_hi = i_lo + 1;
    if i_hi < table.len() {
        lerpf(table[i_lo], table[i_hi], t)
    } else {
        table[i_lo]
    }
}

// ─── Chroma Compress Core ──────────────────────────────────────────────────

/// Chroma compress normalization: hue-dependent scale via trigonometric polynomial.
pub fn chroma_compress_norm(cos_hr: f32, sin_hr: f32, chroma_compress_scale: f32) -> f32 {
    let cos_hr2 = 2.0 * cos_hr * cos_hr - 1.0;
    let sin_hr2 = 2.0 * cos_hr * sin_hr;
    let cos_hr3 = 4.0 * cos_hr * cos_hr * cos_hr - 3.0 * cos_hr;
    let sin_hr3 = 3.0 * sin_hr - 4.0 * sin_hr * sin_hr * sin_hr;

    let w = &CHROMA_COMPRESS_WEIGHTS;
    let m = w[0] * cos_hr + w[1] * cos_hr2 + w[2] * cos_hr3
          + w[4] * sin_hr + w[5] * sin_hr2 + w[6] * sin_hr3
          + w[7];

    m * chroma_compress_scale
}

/// Toe function (forward): smooth compression below limit.
pub fn toe_fwd(x: f32, limit: f32, k1_in: f32, k2_in: f32) -> f32 {
    if x > limit { return x; }
    let k2 = k2_in.max(0.001);
    let k1 = (k1_in * k1_in + k2 * k2).sqrt();
    let k3 = (limit + k1) / (limit + k2);
    let minus_b = k3 * x - k1;
    let minus_ac = k2 * k3 * x;
    0.5 * (minus_b + (minus_b * minus_b + 4.0 * minus_ac).sqrt())
}

/// Toe function (inverse).
pub fn toe_inv(x: f32, limit: f32, k1_in: f32, k2_in: f32) -> f32 {
    if x > limit { return x; }
    let k2 = k2_in.max(0.001);
    let k1 = (k1_in * k1_in + k2 * k2).sqrt();
    let k3 = (limit + k1) / (limit + k2);
    (x * x + k1 * x) / (k3 * (x + k2))
}

/// Forward chroma compression: compress colorfulness based on tonemapped lightness.
pub fn chroma_compress_fwd(
    jmh: &F3, j_ts: f32, mnorm: f32,
    rp: &ResolvedCompressionParams, pc: &ChromaCompressParams,
) -> F3 {
    let j = jmh[0];
    let m = jmh[1];
    let h = jmh[2];

    if m == 0.0 {
        return [j_ts, 0.0, h];
    }

    let nj = j_ts / rp.limit_j_max;
    let snj = (1.0 - nj).max(0.0);
    let limit = nj.powf(rp.model_gamma_inv) * rp.reach_max_m / mnorm;

    let mut m_cp = m * (j_ts / j).powf(rp.model_gamma_inv);
    m_cp /= mnorm;
    m_cp = limit - toe_fwd(limit - m_cp, limit - 0.001, snj * pc.sat, (nj * nj + pc.sat_thr).sqrt());
    m_cp = toe_fwd(m_cp, limit, nj * pc.compr, snj);
    m_cp *= mnorm;

    [j_ts, m_cp, h]
}

/// Inverse chroma compression.
pub fn chroma_compress_inv(
    jmh: &F3, j: f32, mnorm: f32,
    rp: &ResolvedCompressionParams, pc: &ChromaCompressParams,
) -> F3 {
    let j_ts = jmh[0];
    let m_cp = jmh[1];
    let h = jmh[2];

    if m_cp == 0.0 {
        return [j, 0.0, h];
    }

    let nj = j_ts / rp.limit_j_max;
    let snj = (1.0 - nj).max(0.0);
    let limit = nj.powf(rp.model_gamma_inv) * rp.reach_max_m / mnorm;

    let mut m = m_cp / mnorm;
    m = toe_inv(m, limit, nj * pc.compr, snj);
    m = limit - toe_inv(limit - m, limit - 0.001, snj * pc.sat, (nj * nj + pc.sat_thr).sqrt());
    m *= mnorm;
    m *= (j_ts / j).powf(-rp.model_gamma_inv);

    [j, m, h]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toe_fwd_inv_roundtrip() {
        for x in [0.0, 0.1, 0.3, 0.5, 0.8, 0.99, 1.5] {
            let fwd = toe_fwd(x, 1.0, 0.5, 0.3);
            let inv = toe_inv(fwd, 1.0, 0.5, 0.3);
            assert!((inv - x).abs() < 0.001, "toe roundtrip at x={x}: fwd={fwd}, inv={inv}");
        }
    }

    #[test]
    fn toe_fwd_above_limit_is_identity() {
        assert!((toe_fwd(2.0, 1.0, 0.5, 0.3) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn chroma_compress_norm_positive() {
        let cc = init_chroma_compress_params(100.0, &init_tonescale_params(100.0));
        // At 0 degrees hue
        let n = chroma_compress_norm(1.0, 0.0, cc.chroma_compress_scale);
        assert!(n > 0.0, "norm should be positive: {n}");
    }

    #[test]
    fn chroma_compress_norm_varies_with_hue() {
        let cc = init_chroma_compress_params(100.0, &init_tonescale_params(100.0));
        let n0 = chroma_compress_norm(1.0, 0.0, cc.chroma_compress_scale);  // 0 degrees
        let n90 = chroma_compress_norm(0.0, 1.0, cc.chroma_compress_scale); // 90 degrees
        assert!((n0 - n90).abs() > 0.1, "norm should vary with hue: {n0} vs {n90}");
    }

    #[test]
    fn grey_chroma_compress_is_identity() {
        let ts = init_tonescale_params(100.0);
        let cc = init_chroma_compress_params(100.0, &ts);
        let p_in = init_jmh_params(&AP0_PRIMS);
        let p_reach = init_jmh_params(&AP1_PRIMS);
        let shared = init_shared_compression_params(100.0, &p_in, &p_reach);

        // Grey pixel: M=0 → should pass through unchanged
        let jmh = [50.0, 0.0, 0.0];
        let rp = resolve_compression_params(0.0, &shared);
        let result = chroma_compress_fwd(&jmh, 50.0, 1.0, &rp, &cc);
        assert!((result[1]).abs() < 1e-6, "grey M should stay 0: {}", result[1]);
    }

    #[test]
    fn chroma_compress_params_reasonable() {
        let ts = init_tonescale_params(100.0);
        let cc = init_chroma_compress_params(100.0, &ts);
        assert!(cc.sat > 0.0, "sat: {}", cc.sat);
        assert!(cc.compr > 0.0, "compr: {}", cc.compr);
        assert!(cc.chroma_compress_scale > 0.0, "scale: {}", cc.chroma_compress_scale);
    }

    #[test]
    fn shared_compression_params_init() {
        let p_in = init_jmh_params(&AP0_PRIMS);
        let p_reach = init_jmh_params(&AP1_PRIMS);
        let shared = init_shared_compression_params(100.0, &p_in, &p_reach);
        assert!(shared.limit_j_max > 0.0, "limit_j_max: {}", shared.limit_j_max);
        assert!(shared.model_gamma_inv > 0.0, "model_gamma_inv: {}", shared.model_gamma_inv);
        assert_eq!(shared.reach_m_table.len(), TABLE_TOTAL_SIZE);
        // All reach M values should be positive
        for (i, &v) in shared.reach_m_table.iter().enumerate() {
            assert!(v >= 0.0, "reach_m_table[{i}] = {v} (should be >= 0)");
        }
    }
}
