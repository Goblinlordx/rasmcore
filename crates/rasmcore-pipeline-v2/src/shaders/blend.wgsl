struct Params { width: u32, height: u32, opacity: f32, mode: u32, }
@group(0) @binding(2) var<uniform> params: Params;

// ─── Hash for dissolve mode ────────────────────────────────────────────────
fn pcg_hash(v: u32) -> u32 {
    var x = v * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

// ─── W3C compositing spec helpers (non-separable blend modes) ──────────────
fn lum(c: vec3<f32>) -> f32 {
    return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

fn clip_color(c: vec3<f32>) -> vec3<f32> {
    let l = lum(c);
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    var out = c;
    if (n < 0.0) {
        let d = l - n;
        if (d > 0.000001) {
            out = vec3(l) + (out - vec3(l)) * l / d;
        } else {
            out = vec3(l);
        }
    }
    let x2 = max(out.r, max(out.g, out.b));
    if (x2 > 1.0) {
        let l2 = lum(out);
        let d2 = x2 - l2;
        if (d2 > 0.000001) {
            out = vec3(l2) + (out - vec3(l2)) * (1.0 - l2) / d2;
        } else {
            out = vec3(l2);
        }
    }
    return out;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    let d = l - lum(c);
    return clip_color(c + vec3(d));
}

fn sat(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

// SetSat: scale the color so its saturation equals s, preserving channel order.
fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let cmin = min(c.r, min(c.g, c.b));
    let cmax = max(c.r, max(c.g, c.b));
    if (cmax - cmin < 0.000001) {
        return vec3(0.0);
    }
    // Scale each channel: (c - cmin) / (cmax - cmin) * s
    let scale = s / (cmax - cmin);
    return (c - vec3(cmin)) * scale;
}

// ─── Per-channel blend function ────────────────────────────────────────────
fn blend_ch(b: f32, mode: u32) -> f32 {
    switch (mode) {
        case 1u:  { return b * b; }                                                             // multiply
        case 2u:  { return 1.0 - (1.0 - b) * (1.0 - b); }                                     // screen
        case 3u:  { if (b < 0.5) { return 2.0 * b * b; } else { return 1.0 - 2.0 * (1.0 - b) * (1.0 - b); } }  // overlay
        case 4u:  { if (b < 0.5) { return b * (b + 0.5); } else { return 1.0 - (1.0 - b) * (1.5 - b); } }      // soft_light
        case 5u:  { if (b < 0.5) { return 2.0 * b * b; } else { return 1.0 - 2.0 * (1.0 - b) * (1.0 - b); } }  // hard_light
        case 6u:  { return min(b / (1.0 - b + 0.000001), 1.0); }                               // color_dodge
        case 7u:  { return max(1.0 - (1.0 - b) / (b + 0.000001), 0.0); }                       // color_burn
        case 8u:  { return b; }                                                                  // darken
        case 9u:  { return b; }                                                                  // lighten
        case 10u: { return 0.0; }                                                                // difference
        case 11u: { return b + b - 2.0 * b * b; }                                               // exclusion
        case 12u: { return max(b + b - 1.0, 0.0); }                                             // linear_burn
        case 13u: { return min(b + b, 1.0); }                                                   // linear_dodge
        // ── New per-channel modes ──
        case 14u: {                                                                              // vivid_light
            if (b <= 0.5) {
                if (b > 0.0) { return max(1.0 - (1.0 - b) / (2.0 * b), 0.0); } else { return 0.0; }
            } else {
                return min(b / (2.0 * (1.0 - b) + 0.000001), 1.0);
            }
        }
        case 15u: { return clamp(2.0 * b + b - 1.0, 0.0, 1.0); }                               // linear_light
        case 16u: {                                                                              // pin_light
            if (b < 0.5) { return min(b, 2.0 * b); } else { return max(b, 2.0 * b - 1.0); }
        }
        case 17u: { if (b + b >= 1.0) { return 1.0; } else { return 0.0; } }                   // hard_mix
        default:  { return b; }                                                                  // normal / fallback
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    let c = clamp(p, vec4(0.0), vec4(1.0));

    // ── Dissolve (mode 18) ─────────────────────────────────────────────
    if (params.mode == 18u) {
        let h = pcg_hash(gid.y * params.width + gid.x);
        let threshold = f32(h) / 4294967295.0;
        if (threshold > params.opacity) {
            store_pixel(idx, p);
        } else {
            store_pixel(idx, p); // self-blend: always identity
        }
        return;
    }

    // ── Darker/Lighter color (modes 19-20) ─────────────────────────────
    if (params.mode == 19u || params.mode == 20u) {
        // For self-blend: identity. Opacity lerp with itself = identity.
        let result = mix(p, p, params.opacity);
        store_pixel(idx, vec4(result.xyz, p.w));
        return;
    }

    // ── HSL modes (21-24) ──────────────────────────────────────────────
    if (params.mode >= 21u && params.mode <= 24u) {
        let base = c.rgb;
        let blend_color = c.rgb; // self-blend
        var result_rgb: vec3<f32>;
        switch (params.mode) {
            case 21u: { // hue
                result_rgb = set_lum(set_sat(blend_color, sat(base)), lum(base));
            }
            case 22u: { // saturation
                result_rgb = set_lum(set_sat(base, sat(blend_color)), lum(base));
            }
            case 23u: { // color
                result_rgb = set_lum(blend_color, lum(base));
            }
            case 24u: { // luminosity
                result_rgb = set_lum(base, lum(blend_color));
            }
            default: { result_rgb = base; }
        }
        let final_rgb = mix(p.rgb, result_rgb, params.opacity);
        store_pixel(idx, vec4(final_rgb, p.w));
        return;
    }

    // ── Per-channel modes (0-17) ───────────────────────────────────────
    let blended = vec4(blend_ch(c.x, params.mode), blend_ch(c.y, params.mode), blend_ch(c.z, params.mode), c.w);
    let result = mix(p, blended, params.opacity);
    store_pixel(idx, vec4(result.xyz, p.w));
}
