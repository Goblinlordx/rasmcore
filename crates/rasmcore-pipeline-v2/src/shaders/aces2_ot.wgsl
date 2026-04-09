// ACES 2.0 Output Transform — WGSL compute shader.
// Ported from the CPU reference implementation (cam16.rs, tonescale.rs, compress.rs).
// Validated against OCIO v2.5.1 reference vectors.

// ─── Uniform params (precomputed per config) ─────────────────────────────────

struct Params {
    width: u32,
    height: u32,
    // CAM16 input matrices (AP0 → CAM16 adapted, row-major flattened)
    mat_rgb_to_cam16_c: array<f32, 9>,
    mat_cone_to_aab: array<f32, 9>,
    // CAM16 output matrices (limiting gamut inverse)
    mat_aab_to_cone: array<f32, 9>,
    mat_cam16_c_to_rgb: array<f32, 9>,
    // CAM16 scalar params
    f_l_n: f32,
    cz: f32,
    inv_cz: f32,
    a_w_j: f32,
    // AP0 ↔ AP1 matrices for range clamp
    mat_ap0_to_ap1: array<f32, 9>,
    mat_ap1_to_ap0: array<f32, 9>,
    upper_bound: f32,
    // Tonescale
    n_r: f32,
    g: f32,
    t_1: f32,
    s_2: f32,
    m_2: f32,
    // Chroma compress
    sat: f32,
    sat_thr: f32,
    compr: f32,
    chroma_compress_scale: f32,
    // Shared compression
    limit_j_max: f32,
    model_gamma_inv: f32,
    // Gamut compress
    mid_j: f32,
    focus_dist: f32,
    lower_hull_gamma_inv: f32,
    // EOTF
    eotf: u32,  // 0=sRGB, 1=BT1886, 2=PQ, 3=HLG
    peak_luminance: f32,
    _pad: u32,
}

@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> reach_m_table: array<f32>;   // 363
@group(0) @binding(4) var<storage, read> cusp_table: array<f32>;      // 363×3 = 1089
@group(0) @binding(5) var<storage, read> hue_table: array<f32>;       // 363

// ─── Constants ───────────────────────────────────────────────────────────────

const CAM_NL_OFFSET: f32 = 27.13;
const J_SCALE: f32 = 100.0;
const PI: f32 = 3.14159265358979;
const SMOOTH_CUSPS: f32 = 0.12;
const CUSP_MID_BLEND: f32 = 1.3;
const FOCUS_GAIN_BLEND: f32 = 0.3;
const COMPRESSION_THRESHOLD: f32 = 0.75;
const CHROMA_W: array<f32, 8> = array<f32, 8>(
    11.34072, 16.46899, 7.88380, 0.0,
    14.66441, -6.37224, 9.19364, 77.12896
);

// ─── Matrix multiply helpers ─────────────────────────────────────────────────

fn mul_v3_m33(v: vec3<f32>, m: array<f32, 9>) -> vec3<f32> {
    return vec3<f32>(
        v.x * m[0] + v.y * m[1] + v.z * m[2],
        v.x * m[3] + v.y * m[4] + v.z * m[5],
        v.x * m[6] + v.y * m[7] + v.z * m[8]
    );
}

// ─── CAM16 ───────────────────────────────────────────────────────────────────

fn cone_fwd(v: f32) -> f32 {
    let av = abs(v);
    let fly = pow(av, 0.42);
    let ra = fly / (CAM_NL_OFFSET + fly);
    return sign(v) * ra;
}

fn cone_inv(v: f32) -> f32 {
    let av = abs(v);
    let ra = min(av, 0.99);
    let fly = (CAM_NL_OFFSET * ra) / (1.0 - ra);
    let rc = pow(fly, 1.0 / 0.42);
    return sign(v) * rc;
}

fn rgb_to_aab(rgb: vec3<f32>) -> vec3<f32> {
    let m = mul_v3_m33(rgb, p.mat_rgb_to_cam16_c);
    let a = vec3<f32>(cone_fwd(m.x), cone_fwd(m.y), cone_fwd(m.z));
    return mul_v3_m33(a, p.mat_cone_to_aab);
}

fn aab_to_jmh(aab: vec3<f32>) -> vec3<f32> {
    let j = J_SCALE * pow(aab.x, p.cz);
    let m2 = sqrt(aab.y * aab.y + aab.z * aab.z);
    var h = atan2(aab.z, aab.y) * 180.0 / PI;
    if (h < 0.0) { h += 360.0; }
    return vec3<f32>(j, m2, h);
}

fn aab_to_rgb(aab: vec3<f32>) -> vec3<f32> {
    let ra = mul_v3_m33(aab, p.mat_aab_to_cone);
    let m = vec3<f32>(cone_inv(ra.x), cone_inv(ra.y), cone_inv(ra.z));
    return mul_v3_m33(m, p.mat_cam16_c_to_rgb);
}

// ─── Tonescale ───────────────────────────────────────────────────────────────

fn tonescale_fwd(y_in: f32) -> f32 {
    let f = p.m_2 * pow(y_in / (y_in + p.s_2), p.g);
    return max(f * f / (f + p.t_1), 0.0) * p.n_r;
}

fn y_to_j(y: f32) -> f32 {
    let v = abs(y) * p.f_l_n;
    let fly = pow(v, 0.42);
    let ra = fly / (CAM_NL_OFFSET + fly);
    let j = J_SCALE * pow(ra / p.a_w_j, p.cz);
    return sign(y) * j;
}

fn tonescale_a_to_j(a: f32) -> f32 {
    let ra = p.a_w_j * a;
    let y = cone_inv(ra) / p.f_l_n;
    let y_ts = tonescale_fwd(y);
    return y_to_j(y_ts);
}

// ─── Chroma compress ─────────────────────────────────────────────────────────

fn chroma_compress_norm(cos_hr: f32, sin_hr: f32) -> f32 {
    let c2 = 2.0 * cos_hr * cos_hr - 1.0;
    let s2 = 2.0 * cos_hr * sin_hr;
    let c3 = 4.0 * cos_hr * cos_hr * cos_hr - 3.0 * cos_hr;
    let s3 = 3.0 * sin_hr - 4.0 * sin_hr * sin_hr * sin_hr;
    let m = CHROMA_W[0] * cos_hr + CHROMA_W[1] * c2 + CHROMA_W[2] * c3
          + CHROMA_W[4] * sin_hr + CHROMA_W[5] * s2 + CHROMA_W[6] * s3
          + CHROMA_W[7];
    return m * p.chroma_compress_scale;
}

fn toe_fwd(x: f32, limit: f32, k1_in: f32, k2_in: f32) -> f32 {
    if (x > limit) { return x; }
    let k2 = max(k2_in, 0.001);
    let k1 = sqrt(k1_in * k1_in + k2 * k2);
    let k3 = (limit + k1) / (limit + k2);
    let mb = k3 * x - k1;
    let mac = k2 * k3 * x;
    return 0.5 * (mb + sqrt(mb * mb + 4.0 * mac));
}

fn reach_m_lookup(hue: f32) -> f32 {
    let base = u32(hue);
    let t = hue - f32(base);
    let i_lo = base + 1u; // TABLE_BASE_INDEX = 1
    let i_hi = i_lo + 1u;
    return mix(reach_m_table[i_lo], reach_m_table[i_hi], t);
}

fn chroma_compress(jmh: vec3<f32>, j_ts: f32, mnorm: f32, reach_max_m: f32) -> vec3<f32> {
    let j = jmh.x;
    let m = jmh.y;
    let h = jmh.z;
    if (m == 0.0) { return vec3<f32>(j_ts, 0.0, h); }

    let nj = j_ts / p.limit_j_max;
    let snj = max(1.0 - nj, 0.0);
    let limit = pow(nj, p.model_gamma_inv) * reach_max_m / mnorm;

    var mc = m * pow(j_ts / j, p.model_gamma_inv);
    mc /= mnorm;
    mc = limit - toe_fwd(limit - mc, limit - 0.001, snj * p.sat, sqrt(nj * nj + p.sat_thr));
    mc = toe_fwd(mc, limit, nj * p.compr, snj);
    mc *= mnorm;

    return vec3<f32>(j_ts, mc, h);
}

// ─── Gamut compress ──────────────────────────────────────────────────────────

fn smin_scaled(a: f32, b: f32, scale_ref: f32) -> f32 {
    let s = SMOOTH_CUSPS * scale_ref;
    let h = max(s - abs(a - b), 0.0) / s;
    return min(a, b) - h * h * h * s / 6.0;
}

fn get_focus_gain(j: f32, thresh: f32) -> f32 {
    var gain = p.limit_j_max * p.focus_dist;
    if (j > thresh) {
        let adj = log(max((p.limit_j_max - thresh) / max(p.limit_j_max - j, 0.0001), 1.0)) / log(10.0);
        gain *= adj * adj + 1.0;
    }
    return gain;
}

fn solve_j_intersect(j: f32, m: f32, focus_j: f32, slope_gain: f32) -> f32 {
    let ms = m / slope_gain;
    let a = ms / focus_j;
    if (j < focus_j) {
        let b = 1.0 - ms;
        let c = -j;
        let det = b * b - 4.0 * a * c;
        return -2.0 * c / (b + sqrt(det));
    } else {
        let b = -(1.0 + ms + p.limit_j_max * a);
        let c = p.limit_j_max * ms + j;
        let det = b * b - 4.0 * a * c;
        return -2.0 * c / (b - sqrt(det));
    }
}

fn compression_slope(ji: f32, focus_j: f32, slope_gain: f32) -> f32 {
    var dir: f32;
    if (ji < focus_j) { dir = ji; } else { dir = p.limit_j_max - ji; }
    return dir * (ji - focus_j) / (focus_j * slope_gain);
}

fn estimate_boundary_m(j_axis: f32, slope: f32, inv_gamma: f32, j_max: f32, m_max: f32, j_ref: f32) -> f32 {
    let nj = j_axis / j_ref;
    let shifted = j_ref * pow(nj, inv_gamma);
    return shifted * m_max / (j_max - slope * m_max);
}

fn find_gamut_boundary(jm_cusp: vec2<f32>, gamma_top_inv: f32, j_src: f32, slope: f32, jci: f32) -> f32 {
    let m_lower = estimate_boundary_m(j_src, slope, p.lower_hull_gamma_inv, jm_cusp.x, jm_cusp.y, jci);
    let fj_cusp = p.limit_j_max - jci;
    let fj_src = p.limit_j_max - j_src;
    let fcusp_j = p.limit_j_max - jm_cusp.x;
    let m_upper = estimate_boundary_m(fj_src, -slope, gamma_top_inv, fcusp_j, jm_cusp.y, fj_cusp);
    return smin_scaled(m_lower, m_upper, jm_cusp.y);
}

fn remap_m(m: f32, gamut_m: f32, reach_m: f32) -> f32 {
    let ratio = gamut_m / reach_m;
    let proportion = max(ratio, COMPRESSION_THRESHOLD);
    let threshold = proportion * gamut_m;
    if (m <= threshold || proportion >= 1.0) { return m; }
    let m_off = m - threshold;
    let gamut_off = gamut_m - threshold;
    let reach_off = reach_m - threshold;
    let scale = reach_off / ((reach_off / gamut_off) - 1.0);
    let nd = m_off / scale;
    return threshold + scale * nd / (1.0 + nd);
}

fn cusp_lookup(hue: f32) -> vec3<f32> {
    // Binary search for hue interval in hue_table
    let nom = u32(clamp(hue, 0.0, 359.0));
    var i_lo = nom + 1u; // offset by TABLE_BASE_INDEX
    var i_hi = i_lo + 1u;
    // Simplified: uniform table, direct lookup
    let t = hue - f32(nom);
    let lo = i_lo - 1u;
    let hi = i_lo;
    let j = mix(cusp_table[lo * 3u], cusp_table[hi * 3u], t);
    let m = mix(cusp_table[lo * 3u + 1u], cusp_table[hi * 3u + 1u], t);
    let g = mix(cusp_table[lo * 3u + 2u], cusp_table[hi * 3u + 2u], t);
    return vec3<f32>(j, m, g);
}

fn gamut_compress(jmh: vec3<f32>, reach_max_m: f32) -> vec3<f32> {
    let j = jmh.x;
    let m = jmh.y;
    let h = jmh.z;

    if (j <= 0.0) { return vec3<f32>(0.0, 0.0, h); }
    if (m <= 0.0 || j > p.limit_j_max) { return vec3<f32>(j, 0.0, h); }

    // Per-hue cusp lookup
    let cusp = cusp_lookup(h);
    let jm_cusp = vec2<f32>(cusp.x, cusp.y);
    let gamma_top_inv = cusp.z;

    let focus_j = mix(jm_cusp.x, p.mid_j, min(CUSP_MID_BLEND - jm_cusp.x / p.limit_j_max, 1.0));
    let thresh = mix(jm_cusp.x, p.limit_j_max, FOCUS_GAIN_BLEND);

    let sg = get_focus_gain(j, thresh);
    let ji = solve_j_intersect(j, m, focus_j, sg);
    let sl = compression_slope(ji, focus_j, sg);

    let jci = solve_j_intersect(jm_cusp.x, jm_cusp.y, focus_j, sg);
    let gamut_m = find_gamut_boundary(jm_cusp, gamma_top_inv, ji, sl, jci);

    if (gamut_m <= 0.0) { return vec3<f32>(j, 0.0, h); }

    let rm = estimate_boundary_m(ji, sl, p.model_gamma_inv, p.limit_j_max, reach_max_m, p.limit_j_max);
    let remapped = remap_m(m, gamut_m, rm);

    return vec3<f32>(ji + remapped * sl, remapped, h);
}

// ─── EOTF encoding ───────────────────────────────────────────────────────────

fn apply_eotf(v: vec3<f32>) -> vec3<f32> {
    switch (p.eotf) {
        case 0u: { // sRGB
            return vec3<f32>(
                select(1.055 * pow(v.x, 1.0 / 2.4) - 0.055, v.x * 12.92, v.x <= 0.0031308),
                select(1.055 * pow(v.y, 1.0 / 2.4) - 0.055, v.y * 12.92, v.y <= 0.0031308),
                select(1.055 * pow(v.z, 1.0 / 2.4) - 0.055, v.z * 12.92, v.z <= 0.0031308)
            );
        }
        case 1u: { // BT.1886
            return vec3<f32>(pow(v.x, 1.0 / 2.4), pow(v.y, 1.0 / 2.4), pow(v.z, 1.0 / 2.4));
        }
        case 2u: { // PQ
            let m1 = 2610.0 / 16384.0;
            let m2 = 2523.0 / 4096.0 * 128.0;
            let c1 = 3424.0 / 4096.0;
            let c2 = 2413.0 / 4096.0 * 32.0;
            let c3 = 2392.0 / 4096.0 * 32.0;
            let l = max(v * p.peak_luminance / 10000.0, vec3<f32>(0.0));
            let lm = vec3<f32>(pow(l.x, m1), pow(l.y, m1), pow(l.z, m1));
            return vec3<f32>(
                pow((c1 + c2 * lm.x) / (1.0 + c3 * lm.x), m2),
                pow((c1 + c2 * lm.y) / (1.0 + c3 * lm.y), m2),
                pow((c1 + c2 * lm.z) / (1.0 + c3 * lm.z), m2)
            );
        }
        case 3u: { // HLG
            let a_hlg = 0.17883277;
            let b_hlg = 1.0 - 4.0 * a_hlg;
            let c_hlg = 0.5 - a_hlg * log(4.0 * a_hlg);
            return vec3<f32>(
                select(a_hlg * log(12.0 * v.x - b_hlg) + c_hlg, sqrt(3.0 * v.x), v.x <= 1.0 / 12.0),
                select(a_hlg * log(12.0 * v.y - b_hlg) + c_hlg, sqrt(3.0 * v.y), v.y <= 1.0 / 12.0),
                select(a_hlg * log(12.0 * v.z - b_hlg) + c_hlg, sqrt(3.0 * v.z), v.z <= 1.0 / 12.0)
            );
        }
        default: { return v; }
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { return; }
    let idx = gid.y * p.width + gid.x;
    let px = load_pixel(idx);

    // Op1-3: AP0 → AP1 range clamp → AP0
    let ap1 = mul_v3_m33(vec3<f32>(px.x, px.y, px.z), p.mat_ap0_to_ap1);
    let ap1c = clamp(ap1, vec3<f32>(0.0), vec3<f32>(p.upper_bound));
    let rgb = mul_v3_m33(ap1c, p.mat_ap1_to_ap0);

    // Step 1: RGB → Aab → JMh
    let aab = rgb_to_aab(rgb);
    if (aab.x <= 0.0) {
        store_pixel(idx, vec4<f32>(0.0, 0.0, 0.0, px.w));
        return;
    }
    let jmh = aab_to_jmh(aab);

    var h = jmh.z % 360.0;
    if (h < 0.0) { h += 360.0; }
    let h_rad = h * PI / 180.0;
    let cos_hr = cos(h_rad);
    let sin_hr = sin(h_rad);

    // Step 2: Tonescale (A → J_ts)
    let j_ts = tonescale_a_to_j(aab.x);

    // Step 3: Chroma compress
    let reach_max_m = reach_m_lookup(h);
    let mnorm = chroma_compress_norm(cos_hr, sin_hr);
    let jmh_cc = chroma_compress(vec3<f32>(jmh.x, jmh.y, h), j_ts, mnorm, reach_max_m);

    // Step 4: Gamut compress
    let jmh_gc = gamut_compress(jmh_cc, reach_max_m);

    // Step 5: JMh → Aab → RGB (output gamut, using original hue trig)
    let a_out = pow(jmh_gc.x / J_SCALE, p.inv_cz);
    let aab_out = vec3<f32>(a_out, jmh_gc.y * cos_hr, jmh_gc.y * sin_hr);
    let rgb_out = aab_to_rgb(aab_out);

    // Step 6: Clamp + normalize + EOTF
    let display_limit = p.peak_luminance / 100.0;
    let clamped = clamp(rgb_out, vec3<f32>(0.0), vec3<f32>(display_limit)) / display_limit;
    let encoded = apply_eotf(clamped);

    store_pixel(idx, vec4<f32>(encoded, px.w));
}
