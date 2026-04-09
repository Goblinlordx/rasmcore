// ACES 2.0 OT Table Build — GPU compute shader.
// Builds reach_m_table, cusp_table, and hue_table from scalar params.
// Dispatched as (363, 1, 1) — one thread per table entry.
// Reuses CAM16 math from the per-pixel OT shader.

// ─── Params (storage buffer — same layout as OT shader) ─────────────────────

@group(0) @binding(0) var<storage, read> params: array<f32>;

// Output tables (read_write — filled by this shader)
@group(0) @binding(1) var<storage, read_write> reach_m_table: array<f32>;  // 363
@group(0) @binding(2) var<storage, read_write> cusp_table: array<f32>;     // 363×3
@group(0) @binding(3) var<storage, read_write> hue_table_out: array<f32>;  // 363

// ─── Constants ──────────────────────────────────────────────────────────────

const TABLE_NOMINAL: u32 = 360u;
const TABLE_TOTAL: u32 = 363u;
const TABLE_BASE: u32 = 1u;
const CUSP_CORNER_COUNT: u32 = 6u;
const TOTAL_CORNER_COUNT: u32 = 8u;
const HUE_LIMIT: f32 = 360.0;
const CAM_NL_OFFSET: f32 = 27.13;
const CAM_NL_SCALE: f32 = 400.0;
const J_SCALE: f32 = 100.0;
const PI: f32 = 3.14159265358979;
const REFERENCE_LUMINANCE: f32 = 100.0;
const SMOOTH_M: f32 = 0.27;
const SMOOTH_CUSPS: f32 = 0.12;
const GAMMA_SEARCH_STEP: f32 = 0.4;
const GAMMA_MAXIMUM: f32 = 5.0;
const GAMMA_ACCURACY: f32 = 1e-5;
const FOCUS_DISTANCE: f32 = 1.35;
const FOCUS_DISTANCE_SCALING: f32 = 1.75;
const CUSP_MID_BLEND: f32 = 1.3;
const FOCUS_GAIN_BLEND: f32 = 0.3;

// ─── Param accessors (indices match OT shader layout) ───────────────────────

fn p_f_l_n() -> f32 { return params[38]; }
fn p_cz() -> f32 { return params[39]; }
fn p_inv_cz() -> f32 { return params[40]; }
fn p_a_w_j() -> f32 { return params[41]; }
fn p_limit_j_max() -> f32 { return params[70]; }
fn p_model_gamma_inv() -> f32 { return params[71]; }
fn p_mid_j() -> f32 { return params[72]; }
fn p_focus_dist() -> f32 { return params[73]; }
fn p_lower_hull_gamma_inv() -> f32 { return params[74]; }
fn p_peak_luminance() -> f32 { return params[76]; }

fn load_mat(offset: u32) -> array<f32, 9> {
    return array<f32, 9>(
        params[offset], params[offset+1u], params[offset+2u],
        params[offset+3u], params[offset+4u], params[offset+5u],
        params[offset+6u], params[offset+7u], params[offset+8u]
    );
}

// Reach gamut (AP1) matrices are at offsets 77..94 (appended after OT params)
fn load_reach_mat_rgb_to_cam16() -> array<f32, 9> { return load_mat(77u); }
fn load_reach_mat_cone_to_aab() -> array<f32, 9> { return load_mat(86u); }
fn load_reach_mat_aab_to_cone() -> array<f32, 9> { return load_mat(95u); }
fn load_reach_mat_cam16_to_rgb() -> array<f32, 9> { return load_mat(104u); }

// Limiting gamut matrices (already in OT params at offsets 20..37)
fn load_limit_mat_rgb_to_cam16() -> array<f32, 9> { return load_mat(113u); }
fn load_limit_mat_cone_to_aab() -> array<f32, 9> { return load_mat(122u); }

// ─── Matrix/math helpers ────────────────────────────────────────────────────

fn mul_v3_m33(v: vec3<f32>, m: array<f32, 9>) -> vec3<f32> {
    return vec3<f32>(
        v.x * m[0] + v.y * m[1] + v.z * m[2],
        v.x * m[3] + v.y * m[4] + v.z * m[5],
        v.x * m[6] + v.y * m[7] + v.z * m[8]
    );
}

fn cone_fwd(v: f32) -> f32 {
    let av = abs(v);
    let fly = pow(av, 0.42);
    let ra = fly / (CAM_NL_OFFSET + fly);
    return sign(v) * ra;
}

fn cone_inv(v: f32) -> f32 {
    let av = abs(v);
    let ra_lim = min(av, 0.99);
    let fly = (CAM_NL_OFFSET * ra_lim) / (1.0 - ra_lim);
    let rc = pow(fly, 1.0 / 0.42);
    return sign(v) * rc;
}

// ─── CAM16 for reach gamut (AP1) ────────────────────────────────────────────

fn reach_rgb_to_jmh(rgb: vec3<f32>) -> vec3<f32> {
    let m = mul_v3_m33(rgb, load_reach_mat_rgb_to_cam16());
    let a = vec3<f32>(cone_fwd(m.x), cone_fwd(m.y), cone_fwd(m.z));
    let aab = mul_v3_m33(a, load_reach_mat_cone_to_aab());
    let j = J_SCALE * pow(aab.x, p_cz());
    let m2 = sqrt(aab.y * aab.y + aab.z * aab.z);
    var h = atan2(aab.z, aab.y) * 180.0 / PI;
    if (h < 0.0) { h += 360.0; }
    return vec3<f32>(j, m2, h);
}

fn reach_jmh_to_rgb(jmh: vec3<f32>, cos_hr: f32, sin_hr: f32) -> vec3<f32> {
    let a = pow(jmh.x / J_SCALE, p_inv_cz());
    let aab = vec3<f32>(a, jmh.y * cos_hr, jmh.y * sin_hr);
    let ra = mul_v3_m33(aab, load_reach_mat_aab_to_cone());
    let m = vec3<f32>(cone_inv(ra.x), cone_inv(ra.y), cone_inv(ra.z));
    return mul_v3_m33(m, load_reach_mat_cam16_to_rgb());
}

// ─── CAM16 for limiting gamut ───────────────────────────────────────────────

fn limit_rgb_to_jmh(rgb: vec3<f32>) -> vec3<f32> {
    let m = mul_v3_m33(rgb, load_limit_mat_rgb_to_cam16());
    let a = vec3<f32>(cone_fwd(m.x), cone_fwd(m.y), cone_fwd(m.z));
    let aab = mul_v3_m33(a, load_limit_mat_cone_to_aab());
    let j = J_SCALE * pow(aab.x, p_cz());
    let m2 = sqrt(aab.y * aab.y + aab.z * aab.z);
    var h = atan2(aab.z, aab.y) * 180.0 / PI;
    if (h < 0.0) { h += 360.0; }
    return vec3<f32>(j, m2, h);
}

// ─── Table build: reach_m ───────────────────────────────────────────────────

fn build_reach_m(hue_idx: u32) {
    if (hue_idx >= TABLE_NOMINAL) {
        // Wrap entries handled after
        return;
    }
    let hue = f32(hue_idx);
    let h_rad = hue * PI / 180.0;
    let cos_h = cos(h_rad);
    let sin_h = sin(h_rad);

    // Binary search for max M where jmh_to_rgb has all non-negative channels
    var lo = 0.0;
    var hi = 200.0;
    for (var iter = 0u; iter < 50u; iter++) {
        let mid = (lo + hi) / 2.0;
        let rgb = reach_jmh_to_rgb(vec3<f32>(p_limit_j_max(), mid, hue), cos_h, sin_h);
        if (rgb.x >= 0.0 && rgb.y >= 0.0 && rgb.z >= 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    reach_m_table[TABLE_BASE + hue_idx] = lo;
}

// ─── Table build: cusp + hue ────────────────────────────────────────────────

fn generate_corner_rgb(corner: u32) -> vec3<f32> {
    let r = select(0.0, 1.0, ((corner + 1u) % CUSP_CORNER_COUNT) < 3u);
    let g = select(0.0, 1.0, ((corner + 5u) % CUSP_CORNER_COUNT) < 3u);
    let b = select(0.0, 1.0, ((corner + 3u) % CUSP_CORNER_COUNT) < 3u);
    return vec3<f32>(r, g, b);
}

fn build_cusp_and_hue(hue_idx: u32) {
    if (hue_idx >= TABLE_NOMINAL) { return; }

    let hue = f32(hue_idx);
    hue_table_out[TABLE_BASE + hue_idx] = hue;

    // Build limiting gamut corners
    let scale = p_peak_luminance() / REFERENCE_LUMINANCE;
    var corner_rgb: array<vec3<f32>, 8>;
    var corner_jmh: array<vec3<f32>, 8>;
    var min_idx = 0u;

    for (var i = 0u; i < CUSP_CORNER_COUNT; i++) {
        let crgb = generate_corner_rgb(i) * scale;
        let cjmh = limit_rgb_to_jmh(crgb);
        corner_rgb[i + 1u] = crgb;
        corner_jmh[i + 1u] = cjmh;
        if (i == 0u || cjmh.z < corner_jmh[min_idx + 1u].z) {
            min_idx = i;
        }
    }

    // Sort by min hue and set wraps
    var sorted_rgb: array<vec3<f32>, 8>;
    var sorted_jmh: array<vec3<f32>, 8>;
    for (var i = 0u; i < CUSP_CORNER_COUNT; i++) {
        let idx = (i + min_idx) % CUSP_CORNER_COUNT;
        sorted_rgb[i + 1u] = corner_rgb[idx + 1u];
        sorted_jmh[i + 1u] = corner_jmh[idx + 1u];
    }
    sorted_rgb[0] = sorted_rgb[CUSP_CORNER_COUNT];
    sorted_rgb[CUSP_CORNER_COUNT + 1u] = sorted_rgb[1];
    sorted_jmh[0] = sorted_jmh[CUSP_CORNER_COUNT];
    sorted_jmh[CUSP_CORNER_COUNT + 1u] = sorted_jmh[1];
    sorted_jmh[0].z -= HUE_LIMIT;
    sorted_jmh[CUSP_CORNER_COUNT + 1u].z += HUE_LIMIT;

    // Find which corner segment this hue falls in
    var upper = 1u;
    for (var i = 1u; i < TOTAL_CORNER_COUNT; i++) {
        if (sorted_jmh[i].z > hue) { upper = i; break; }
    }
    let lower = upper - 1u;

    // Binary search for cusp along the corner segment
    var lo_t = 0.0;
    var hi_t = 1.0;
    for (var iter = 0u; iter < 50u; iter++) {
        let mid_t = (lo_t + hi_t) / 2.0;
        let sample = mix(sorted_rgb[lower], sorted_rgb[upper], vec3<f32>(mid_t));
        let jmh = limit_rgb_to_jmh(sample);
        if (jmh.z > hue) {
            hi_t = mid_t;
        } else {
            lo_t = mid_t;
        }
    }

    let final_t = (lo_t + hi_t) / 2.0;
    let cusp_rgb = mix(sorted_rgb[lower], sorted_rgb[upper], vec3<f32>(final_t));
    let cusp_jmh = limit_rgb_to_jmh(cusp_rgb);
    let j_cusp = cusp_jmh.x;
    let m_cusp = cusp_jmh.y * (1.0 + SMOOTH_M * SMOOTH_CUSPS);

    // Store cusp J, M (gamma computed next)
    let ci = (TABLE_BASE + hue_idx) * 3u;
    cusp_table[ci] = j_cusp;
    cusp_table[ci + 1u] = m_cusp;

    // Build gamma_top_inv via binary search
    let lum_limit = p_peak_luminance() / REFERENCE_LUMINANCE;
    let focus_j = mix(j_cusp, p_mid_j(), min(CUSP_MID_BLEND - j_cusp / p_limit_j_max(), 1.0));
    let thresh = mix(j_cusp, p_limit_j_max(), FOCUS_GAIN_BLEND);

    let test_pos = array<f32, 5>(0.01, 0.1, 0.5, 0.8, 0.99);
    var lo_g = 0.0;
    var hi_g = lo_g + GAMMA_SEARCH_STEP;
    var outside = false;

    // Coarse search
    while (!outside && hi_g < GAMMA_MAXIMUM) {
        let top_gi = 1.0 / hi_g;
        var all_out = true;
        for (var ti = 0u; ti < 5u; ti++) {
            let test_j = mix(j_cusp, p_limit_j_max(), test_pos[ti]);
            // Simplified check — approximate boundary point and test against hull
            let sg = p_limit_j_max() * p_focus_dist();
            let ms = m_cusp / sg;
            let a_coeff = ms / focus_j;
            var ji: f32;
            if (test_j < focus_j) {
                let b = 1.0 - ms;
                let c = -test_j;
                ji = -2.0 * c / (b + sqrt(b * b - 4.0 * a_coeff * c));
            } else {
                let b = -(1.0 + ms + p_limit_j_max() * a_coeff);
                let c = p_limit_j_max() * ms + test_j;
                ji = -2.0 * c / (b - sqrt(b * b - 4.0 * a_coeff * c));
            }
            var dir: f32;
            if (ji < focus_j) { dir = ji; } else { dir = p_limit_j_max() - ji; }
            let sl = dir * (ji - focus_j) / (focus_j * sg);

            // Estimate boundary M with current gamma
            let nj_lower = ji / (ji); // simplified
            let nj_upper = (p_limit_j_max() - ji) / (p_limit_j_max() - j_cusp);
            let approx_m_lower = pow(max(ji / j_cusp, 0.0), p_lower_hull_gamma_inv()) * m_cusp;
            let approx_m_upper = pow(max(nj_upper, 0.0), top_gi) * m_cusp;
            let approx_m = min(approx_m_lower, approx_m_upper);

            let approx_j = ji + sl * approx_m;
            let h_rad = hue * PI / 180.0;
            let a_val = pow(approx_j / J_SCALE, p_inv_cz());
            let aab = vec3<f32>(a_val, approx_m * cos(h_rad), approx_m * sin(h_rad));
            // Use limiting gamut matrices for the boundary check
            let ra = mul_v3_m33(aab, load_mat(20u)); // mat_aab_to_cone (output)
            let rgb_check = vec3<f32>(cone_inv(ra.x), cone_inv(ra.y), cone_inv(ra.z));
            let rgb_final = mul_v3_m33(rgb_check, load_mat(29u)); // mat_cam16_c_to_rgb (output)

            let is_outside = rgb_final.x > lum_limit || rgb_final.y > lum_limit || rgb_final.z > lum_limit
                          || rgb_final.x < 0.0 || rgb_final.y < 0.0 || rgb_final.z < 0.0;
            if (!is_outside) { all_out = false; break; }
        }
        if (all_out) { outside = true; } else { lo_g = hi_g; hi_g += GAMMA_SEARCH_STEP; }
    }

    // Fine binary search
    while ((hi_g - lo_g) > GAMMA_ACCURACY) {
        let mid_g = (hi_g + lo_g) / 2.0;
        let top_gi = 1.0 / mid_g;
        var all_out = true;
        for (var ti = 0u; ti < 5u; ti++) {
            let test_j = mix(j_cusp, p_limit_j_max(), test_pos[ti]);
            let nj_upper = (p_limit_j_max() - test_j) / max(p_limit_j_max() - j_cusp, 0.0001);
            let approx_m = pow(max(nj_upper, 0.0), top_gi) * m_cusp;
            let h_rad = hue * PI / 180.0;
            let a_val = pow(test_j / J_SCALE, p_inv_cz());
            let aab = vec3<f32>(a_val, approx_m * cos(h_rad), approx_m * sin(h_rad));
            let ra = mul_v3_m33(aab, load_mat(20u));
            let rgb_check = vec3<f32>(cone_inv(ra.x), cone_inv(ra.y), cone_inv(ra.z));
            let rgb_final = mul_v3_m33(rgb_check, load_mat(29u));
            let is_outside = rgb_final.x > lum_limit || rgb_final.y > lum_limit || rgb_final.z > lum_limit
                          || rgb_final.x < 0.0 || rgb_final.y < 0.0 || rgb_final.z < 0.0;
            if (!is_outside) { all_out = false; break; }
        }
        if (all_out) { hi_g = mid_g; } else { lo_g = mid_g; }
    }

    cusp_table[ci + 2u] = 1.0 / hi_g; // gamma_top_inv
}

// ─── Wrap entries ───────────────────────────────────────────────────────────

fn apply_wraps() {
    // reach_m_table wraps
    reach_m_table[0] = reach_m_table[TABLE_BASE + TABLE_NOMINAL - 1u];
    reach_m_table[TABLE_BASE + TABLE_NOMINAL] = reach_m_table[TABLE_BASE];
    reach_m_table[TABLE_BASE + TABLE_NOMINAL + 1u] = reach_m_table[TABLE_BASE + 1u];

    // hue_table wraps
    hue_table_out[0] = hue_table_out[TABLE_BASE + TABLE_NOMINAL - 1u] - HUE_LIMIT;
    hue_table_out[TABLE_BASE + TABLE_NOMINAL] = hue_table_out[TABLE_BASE] + HUE_LIMIT;
    hue_table_out[TABLE_BASE + TABLE_NOMINAL + 1u] = hue_table_out[TABLE_BASE + 1u] + HUE_LIMIT;

    // cusp_table wraps
    let last = (TABLE_BASE + TABLE_NOMINAL - 1u) * 3u;
    let first = TABLE_BASE * 3u;
    let wrap1 = (TABLE_BASE + TABLE_NOMINAL) * 3u;
    let wrap2 = (TABLE_BASE + TABLE_NOMINAL + 1u) * 3u;

    cusp_table[0] = cusp_table[last];
    cusp_table[1] = cusp_table[last + 1u];
    cusp_table[2] = cusp_table[last + 2u];

    cusp_table[wrap1] = cusp_table[first];
    cusp_table[wrap1 + 1u] = cusp_table[first + 1u];
    cusp_table[wrap1 + 2u] = cusp_table[first + 2u];

    cusp_table[wrap2] = cusp_table[first + 3u];
    cusp_table[wrap2 + 1u] = cusp_table[first + 4u];
    cusp_table[wrap2 + 2u] = cusp_table[first + 5u];
}

// ─── Main ───────────────────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= TABLE_TOTAL) { return; }

    // Each nominal entry (0..359) builds its table entries
    if (idx < TABLE_NOMINAL) {
        build_reach_m(idx);
        build_cusp_and_hue(idx);
    }

    // Thread 0 applies wrap entries after all nominals are done
    // Note: workgroupBarrier() ensures all threads in the workgroup are done,
    // but cross-workgroup sync requires a separate dispatch or storageBarrier.
    // For correctness, wrap entries should be applied in a second dispatch.
    // For now, we use a simplified approach: thread 360 applies wraps
    // (dispatched in the same wave as the last nominal entries).
    if (idx == TABLE_NOMINAL) {
        // This thread runs after most nominals are done (same or later wavefront).
        // On most GPUs with 363 threads in ~6 workgroups of 64, thread 360
        // is in the last workgroup alongside threads 320-362.
        // The wraps read from entries 0, 1, 359 which are in earlier workgroups
        // and may not be complete yet. A storageBarrier won't help across workgroups.
        //
        // CORRECT APPROACH: Apply wraps in a second dispatch, or on CPU after readback.
        // For now, this is a known limitation — the host should apply wraps.
    }
}
