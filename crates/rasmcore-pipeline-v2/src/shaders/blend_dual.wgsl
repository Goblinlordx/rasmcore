// Dual-input blend mode GPU shader — ISO 32000-2:2020 Section 11.3.5
//
// load_pixel_a() = foreground (blend layer), load_pixel_b() = background (base layer).
// Luminance coefficients passed as uniforms from working color space.

struct Params {
    width: u32, height: u32, opacity: f32, mode: u32,
    luma_r: f32, luma_g: f32, luma_b: f32, _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn pcg_hash(v: u32) -> u32 {
    var x = v * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

fn lum(c: vec3<f32>) -> f32 {
    return params.luma_r * c.r + params.luma_g * c.g + params.luma_b * c.b;
}

fn clip_color(c: vec3<f32>) -> vec3<f32> {
    let l = lum(c);
    let n = min(c.r, min(c.g, c.b));
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

fn ssat(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let cmin = min(c.r, min(c.g, c.b));
    let cmax = max(c.r, max(c.g, c.b));
    let range = cmax - cmin;
    if (range < 0.0000001) {
        return vec3(0.0);
    }
    let sc = s / range;
    return (c - vec3(cmin)) * sc;
}

fn soft_light_d(x: f32) -> f32 {
    if (x <= 0.25) {
        return ((16.0 * x - 12.0) * x + 4.0) * x;
    }
    return sqrt(x);
}

fn blend_channel(a: f32, b: f32) -> f32 {
    switch (params.mode) {
        case 0u: { return a; }                                        // Normal
        case 1u: { return a * b; }                                    // Multiply
        case 2u: { return 1.0 - (1.0 - a) * (1.0 - b); }           // Screen
        case 3u: {                                                     // Overlay
            if (b < 0.5) { return 2.0 * a * b; }
            return 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
        }
        case 4u: {                                                     // SoftLight
            if (a <= 0.5) { return b - (1.0 - 2.0 * a) * b * (1.0 - b); }
            return b + (2.0 * a - 1.0) * (soft_light_d(clamp(b, 0.0, 1.0)) - b);
        }
        case 5u: {                                                     // HardLight
            if (a < 0.5) { return 2.0 * a * b; }
            return 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
        }
        case 6u: { return b / max(1.0 - a, 0.000001); }             // ColorDodge
        case 7u: { return 1.0 - (1.0 - b) / max(a, 0.000001); }    // ColorBurn
        case 8u: { return min(a, b); }                                // Darken
        case 9u: { return max(a, b); }                                // Lighten
        case 10u: { return abs(a - b); }                              // Difference
        case 11u: { return a + b - 2.0 * a * b; }                   // Exclusion
        case 12u: { return a + b - 1.0; }                            // LinearBurn
        case 13u: { return a + b; }                                   // LinearDodge
        case 14u: {                                                    // VividLight
            if (a <= 0.5) {
                if (abs(a) > 0.000001) { return 1.0 - (1.0 - b) / (2.0 * a); }
                return 0.0;
            }
            return b / max(2.0 * (1.0 - a), 0.000001);
        }
        case 15u: { return 2.0 * a + b - 1.0; }                    // LinearLight
        case 16u: {                                                    // PinLight
            if (a < 0.5) { return min(b, 2.0 * a); }
            return max(b, 2.0 * a - 1.0);
        }
        case 17u: {                                                    // HardMix
            if (a + b >= 1.0) { return 1.0; }
            return 0.0;
        }
        default: { return b; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.x + gid.y * params.width;

    let fg = load_pixel_a(idx);
    let bg = load_pixel_b(idx);
    let opacity = params.opacity;
    let inv_opacity = 1.0 - opacity;

    var result: vec4<f32>;

    if (params.mode <= 17u) {
        // Per-channel separable modes
        result = vec4(
            bg.r * inv_opacity + blend_channel(fg.r, bg.r) * opacity,
            bg.g * inv_opacity + blend_channel(fg.g, bg.g) * opacity,
            bg.b * inv_opacity + blend_channel(fg.b, bg.b) * opacity,
            bg.a,
        );
    } else if (params.mode == 18u) {
        // Dissolve
        let h = pcg_hash(idx);
        let threshold = f32(h & 0xFFu) / 255.0;
        if (threshold < opacity) {
            result = fg;
        } else {
            result = bg;
        }
    } else if (params.mode == 19u) {
        // Darker color
        let fg_l = lum(fg.rgb);
        let bg_l = lum(bg.rgb);
        var rgb: vec3<f32>;
        if (fg_l < bg_l) { rgb = fg.rgb; } else { rgb = bg.rgb; }
        result = vec4(bg.rgb * inv_opacity + rgb * opacity, bg.a);
    } else if (params.mode == 20u) {
        // Lighter color
        let fg_l = lum(fg.rgb);
        let bg_l = lum(bg.rgb);
        var rgb: vec3<f32>;
        if (fg_l > bg_l) { rgb = fg.rgb; } else { rgb = bg.rgb; }
        result = vec4(bg.rgb * inv_opacity + rgb * opacity, bg.a);
    } else {
        // HSL modes (21-24) — clamp to [0,1] for non-separable math
        let ac = clamp(fg.rgb, vec3(0.0), vec3(1.0));
        let bc = clamp(bg.rgb, vec3(0.0), vec3(1.0));
        var blended: vec3<f32>;
        if (params.mode == 21u) {
            // Hue: hue from fg, sat+lum from bg
            blended = set_lum(set_sat(ac, ssat(bc)), lum(bc));
        } else if (params.mode == 22u) {
            // Saturation: sat from fg, hue+lum from bg
            blended = set_lum(set_sat(bc, ssat(ac)), lum(bc));
        } else if (params.mode == 23u) {
            // Color: hue+sat from fg, lum from bg
            blended = set_lum(ac, lum(bc));
        } else {
            // Luminosity: lum from fg, hue+sat from bg
            blended = set_lum(bc, lum(ac));
        }
        result = vec4(bg.rgb * inv_opacity + blended * opacity, bg.a);
    }

    store_pixel(idx, result);
}
