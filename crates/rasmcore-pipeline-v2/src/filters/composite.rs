//! Alpha and compositing filters — premultiply, unpremultiply, blend modes, blend-if.
//!
//! All filters operate on f32 RGBA data. GPU shaders provided for all.

use crate::node::PipelineError;
use crate::ops::Filter;

// ─── Premultiply ───────────────────────────────────────────────────────────

/// Premultiply alpha — multiply RGB channels by the alpha channel.
///
/// `out.rgb = in.rgb * in.a`
/// `out.a = in.a`
///
/// Required before alpha compositing for correct blending math.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "premultiply", category = "composite", cost = "O(n)")]
pub struct Premultiply;

impl Filter for Premultiply {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            let a = px[3];
            px[0] *= a;
            px[1] *= a;
            px[2] *= a;
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(PREMULTIPLY_WGSL)
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        Some(buf)
    }
}

const PREMULTIPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    store_pixel(idx, vec4(p.rgb * p.a, p.a));
}
"#;

// ─── Unpremultiply ─────────────────────────────────────────────────────────

/// Unpremultiply alpha — divide RGB channels by the alpha channel.
///
/// `out.rgb = in.rgb / in.a` (if a > 0, else 0)
/// `out.a = in.a`
///
/// Reverses premultiplication after compositing.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "unpremultiply", category = "composite", cost = "O(n)")]
pub struct Unpremultiply;

impl Filter for Unpremultiply {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            // Branchless: use select(a > eps, 1/a, 0) to avoid branch in the hot loop.
            // LLVM can vectorize this to a masked divide.
            let a = px[3];
            let inv_a = if a > 1e-7 { 1.0 / a } else { 0.0 };
            px[0] *= inv_a;
            px[1] *= inv_a;
            px[2] *= inv_a;
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(UNPREMULTIPLY_WGSL)
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        Some(buf)
    }
}

const UNPREMULTIPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    if (p.a > 0.0000001) {
        store_pixel(idx, vec4(p.rgb / p.a, p.a));
    } else {
        store_pixel(idx, p);
    }
}
"#;

// ─── Blend ─────────────────────────────────────────────────────────────────

/// Multi-mode blend — blends the input toward a computed result using standard
/// Photoshop/DaVinci blend modes.
///
/// `opacity` controls the mix between original and blended result.
/// This is a single-input filter: it applies the blend mode to the input
/// against itself (useful for self-blending effects like "multiply to darken").
///
/// For two-layer compositing, use the pipeline's composite node instead.
///
/// Modes:
///   0=normal, 1=multiply, 2=screen, 3=overlay, 4=soft_light,
///   5=hard_light, 6=color_dodge, 7=color_burn, 8=darken, 9=lighten,
///   10=difference, 11=exclusion, 12=linear_burn, 13=linear_dodge,
///   14=vivid_light, 15=linear_light, 16=pin_light, 17=hard_mix,
///   18=dissolve, 19=darker_color, 20=lighter_color,
///   21=hue, 22=saturation, 23=color, 24=luminosity
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "blend", category = "composite", cost = "O(n)")]
pub struct Blend {
    /// Blend mode (0-24, see filter docs for mode list)
    #[param(min = 0, max = 24, step = 1, default = 1)]
    pub mode: u32,
    /// Blend opacity (0 = original, 1 = fully blended)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub opacity: f32,
}

// ─── W3C compositing spec helpers (non-separable blend modes) ──────────────

fn css_lum(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

fn clip_color(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let l = css_lum(r, g, b);
    let n = r.min(g).min(b);
    let _x = r.max(g).max(b);
    let (mut ro, mut go, mut bo) = (r, g, b);
    if n < 0.0 {
        let d = (l - n).max(1e-7);
        ro = l + (ro - l) * l / d;
        go = l + (go - l) * l / d;
        bo = l + (bo - l) * l / d;
    }
    let x2 = ro.max(go).max(bo);
    if x2 > 1.0 {
        let l2 = css_lum(ro, go, bo);
        let d2 = (x2 - l2).max(1e-7);
        ro = l2 + (ro - l2) * (1.0 - l2) / d2;
        go = l2 + (go - l2) * (1.0 - l2) / d2;
        bo = l2 + (bo - l2) * (1.0 - l2) / d2;
    }
    (ro, go, bo)
}

fn set_lum(r: f32, g: f32, b: f32, l: f32) -> (f32, f32, f32) {
    let d = l - css_lum(r, g, b);
    clip_color(r + d, g + d, b + d)
}

fn css_sat(r: f32, g: f32, b: f32) -> f32 {
    r.max(g).max(b) - r.min(g).min(b)
}

fn set_sat(r: f32, g: f32, b: f32, s: f32) -> (f32, f32, f32) {
    let cmin = r.min(g).min(b);
    let cmax = r.max(g).max(b);
    let range = cmax - cmin;
    if range < 1e-7 {
        return (0.0, 0.0, 0.0);
    }
    let scale = s / range;
    ((r - cmin) * scale, (g - cmin) * scale, (b - cmin) * scale)
}

// Simple hash for dissolve mode.
fn pcg_hash(v: u32) -> u32 {
    let mut x = v.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    (x >> 22) ^ x
}

impl Filter for Blend {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let opacity = self.opacity;
        let inv_opacity = 1.0 - opacity;
        let mut out = input.to_vec();

        match self.mode {
            // ── Per-channel modes (0-17) ───────────────────────────────
            0..=17 => {
                // Hoist mode dispatch outside the loop so the inner loop is a
                // single arithmetic expression that LLVM can auto-vectorize.
                let blend_fn: fn(f32) -> f32 = match self.mode {
                    1 => |b: f32| b * b,
                    2 => |b: f32| 1.0 - (1.0 - b) * (1.0 - b),
                    3 => |b: f32| if b < 0.5 { 2.0 * b * b } else { 1.0 - 2.0 * (1.0 - b) * (1.0 - b) },
                    4 => |b: f32| if b < 0.5 { b * (b + 0.5) } else { 1.0 - (1.0 - b) * (1.5 - b) },
                    5 => |b: f32| if b < 0.5 { 2.0 * b * b } else { 1.0 - 2.0 * (1.0 - b) * (1.0 - b) },
                    6 => |b: f32| if b < 1.0 { (b / (1.0 - b + 1e-6)).min(1.0) } else { 1.0 },
                    7 => |b: f32| if b > 0.0 { 1.0 - ((1.0 - b) / (b + 1e-6)).min(1.0) } else { 0.0 },
                    8 => |b: f32| b,    // darken (identity for self-blend)
                    9 => |b: f32| b,    // lighten (identity for self-blend)
                    10 => |_: f32| 0.0, // difference (zero for self-blend)
                    11 => |b: f32| b + b - 2.0 * b * b,
                    12 => |b: f32| (b + b - 1.0).max(0.0),
                    13 => |b: f32| (b + b).min(1.0),
                    14 => |b: f32| { // vivid_light
                        if b <= 0.5 {
                            if b > 0.0 { (1.0 - (1.0 - b) / (2.0 * b)).max(0.0) } else { 0.0 }
                        } else {
                            (b / (2.0 * (1.0 - b) + 1e-6)).min(1.0)
                        }
                    },
                    15 => |b: f32| (2.0 * b + b - 1.0).clamp(0.0, 1.0), // linear_light
                    16 => |b: f32| { // pin_light
                        if b < 0.5 { b.min(2.0 * b) } else { b.max(2.0 * b - 1.0) }
                    },
                    17 => |b: f32| if b + b >= 1.0 { 1.0 } else { 0.0 }, // hard_mix
                    _ => |b: f32| b,
                };

                for px in out.chunks_exact_mut(4) {
                    let r = px[0].clamp(0.0, 1.0);
                    let g = px[1].clamp(0.0, 1.0);
                    let b = px[2].clamp(0.0, 1.0);
                    px[0] = px[0] * inv_opacity + blend_fn(r) * opacity;
                    px[1] = px[1] * inv_opacity + blend_fn(g) * opacity;
                    px[2] = px[2] * inv_opacity + blend_fn(b) * opacity;
                }
            }

            // ── Dissolve (18) ──────────────────────────────────────────
            18 => {
                for (i, _px) in out.chunks_exact_mut(4).enumerate() {
                    let h = pcg_hash(i as u32);
                    let threshold = (h as f64 / u32::MAX as f64) as f32;
                    if threshold > opacity {
                        // keep original — no change needed
                    }
                    // else: use blended pixel (identity for self-blend)
                }
            }

            // ── Darker color (19) ──────────────────────────────────────
            // Compare luminance of base vs blend pixel, keep the darker.
            // For self-blend: identity (same pixel both sides).
            19 | 20 => {
                // No-op for self-blend. Opacity lerp with itself = identity.
            }

            // ── HSL modes (21-24) ──────────────────────────────────────
            21..=24 => {
                for px in out.chunks_exact_mut(4) {
                    let (r, g, b) = (px[0].clamp(0.0, 1.0), px[1].clamp(0.0, 1.0), px[2].clamp(0.0, 1.0));
                    // For self-blend, base == blend. Formulas still applied for correctness.
                    let (ro, go, bo) = match self.mode {
                        21 => { // hue: hue from blend, sat+lum from base
                            let (sr, sg, sb) = set_sat(r, g, b, css_sat(r, g, b));
                            set_lum(sr, sg, sb, css_lum(r, g, b))
                        }
                        22 => { // saturation: sat from blend, hue+lum from base
                            let (sr, sg, sb) = set_sat(r, g, b, css_sat(r, g, b));
                            set_lum(sr, sg, sb, css_lum(r, g, b))
                        }
                        23 => { // color: hue+sat from blend, lum from base
                            set_lum(r, g, b, css_lum(r, g, b))
                        }
                        24 => { // luminosity: lum from blend, hue+sat from base
                            set_lum(r, g, b, css_lum(r, g, b))
                        }
                        _ => (r, g, b),
                    };
                    px[0] = px[0] * inv_opacity + ro * opacity;
                    px[1] = px[1] * inv_opacity + go * opacity;
                    px[2] = px[2] * inv_opacity + bo * opacity;
                }
            }

            _ => {} // unknown mode: identity
        }

        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(include_str!("../shaders/blend.wgsl"))
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.opacity.to_le_bytes());
        buf.extend_from_slice(&self.mode.to_le_bytes());
        Some(buf)
    }
}

// ─── Blend If ──────────────────────────────────────────────────────────────

/// Photoshop-style "Blend If" — conditional blending based on luminance.
///
/// Pixels with luminance below `shadow_threshold` or above `highlight_threshold`
/// are faded to transparent (alpha reduced). Creates smooth falloff in the
/// transition zones.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "blend_if", category = "composite", cost = "O(n)")]
pub struct BlendIf {
    /// Shadow threshold — pixels darker than this start fading
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.0)]
    pub shadow_threshold: f32,
    /// Shadow feather — transition width for shadow fade
    #[param(min = 0.0, max = 0.5, step = 0.02, default = 0.1)]
    pub shadow_feather: f32,
    /// Highlight threshold — pixels brighter than this start fading
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 1.0)]
    pub highlight_threshold: f32,
    /// Highlight feather — transition width for highlight fade
    #[param(min = 0.0, max = 0.5, step = 0.02, default = 0.1)]
    pub highlight_feather: f32,
}

fn bt709_luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

impl Filter for BlendIf {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let s_lo = self.shadow_threshold;
        let s_hi = s_lo + self.shadow_feather.max(1e-6);
        let h_hi = self.highlight_threshold;
        let h_lo = h_hi - self.highlight_feather.max(1e-6);

        for px in out.chunks_exact_mut(4) {
            let luma = bt709_luma(px[0], px[1], px[2]);
            // Shadow fade: 0 at s_lo, 1 at s_hi
            let shadow_mix = ((luma - s_lo) / (s_hi - s_lo)).clamp(0.0, 1.0);
            // Highlight fade: 1 at h_lo, 0 at h_hi
            let highlight_mix = ((h_hi - luma) / (h_hi - h_lo)).clamp(0.0, 1.0);
            let mask = shadow_mix * highlight_mix;
            px[3] *= mask;
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(BLEND_IF_WGSL)
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.shadow_threshold.to_le_bytes());
        buf.extend_from_slice(&self.shadow_feather.max(1e-6).to_le_bytes());
        buf.extend_from_slice(&self.highlight_threshold.to_le_bytes());
        buf.extend_from_slice(&self.highlight_feather.max(1e-6).to_le_bytes());
        Some(buf)
    }
}

const BLEND_IF_WGSL: &str = r#"
struct Params {
    width: u32, height: u32,
    shadow_threshold: f32, shadow_feather: f32,
    highlight_threshold: f32, highlight_feather: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    let luma = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
    let s_lo = params.shadow_threshold;
    let s_hi = s_lo + params.shadow_feather;
    let h_hi = params.highlight_threshold;
    let h_lo = h_hi - params.highlight_feather;
    let shadow_mix = clamp((luma - s_lo) / (s_hi - s_lo), 0.0, 1.0);
    let highlight_mix = clamp((h_hi - luma) / (h_hi - h_lo), 0.0, 1.0);
    let mask = shadow_mix * highlight_mix;
    store_pixel(idx, vec4(p.xyz, p.w * mask));
}
"#;

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
        vec![r, g, b, a]
    }

    #[test]
    fn premultiply_basic() {
        let input = pixel(0.8, 0.6, 0.4, 0.5);
        let out = Premultiply.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.4).abs() < 1e-6);
        assert!((out[1] - 0.3).abs() < 1e-6);
        assert!((out[2] - 0.2).abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn unpremultiply_basic() {
        let input = pixel(0.4, 0.3, 0.2, 0.5);
        let out = Unpremultiply.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.8).abs() < 1e-6);
        assert!((out[1] - 0.6).abs() < 1e-6);
        assert!((out[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn unpremultiply_zero_alpha() {
        let input = pixel(0.0, 0.0, 0.0, 0.0);
        let out = Unpremultiply.compute(&input, 1, 1).unwrap();
        assert_eq!(out, input); // no division by zero
    }

    #[test]
    fn premultiply_unpremultiply_roundtrip() {
        let input = pixel(0.7, 0.3, 0.9, 0.6);
        let pre = Premultiply.compute(&input, 1, 1).unwrap();
        let back = Unpremultiply.compute(&pre, 1, 1).unwrap();
        for i in 0..4 {
            assert!((back[i] - input[i]).abs() < 1e-5, "channel {i}: {} vs {}", back[i], input[i]);
        }
    }

    #[test]
    fn blend_multiply_darkens() {
        let input = pixel(0.8, 0.8, 0.8, 1.0);
        let f = Blend { mode: 1, opacity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] < 0.8, "multiply should darken: {}", out[0]);
        assert!((out[0] - 0.64).abs() < 1e-5); // 0.8 * 0.8
    }

    #[test]
    fn blend_screen_brightens() {
        let input = pixel(0.5, 0.5, 0.5, 1.0);
        let f = Blend { mode: 2, opacity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] > 0.5, "screen should brighten: {}", out[0]);
        assert!((out[0] - 0.75).abs() < 1e-5); // 1 - (1-0.5)*(1-0.5)
    }

    #[test]
    fn blend_opacity_zero_is_identity() {
        let input = pixel(0.5, 0.3, 0.7, 1.0);
        let f = Blend { mode: 1, opacity: 0.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        for i in 0..3 {
            assert!((out[i] - input[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn blend_overlay_midtone() {
        // overlay at 0.5 => 2*0.5*0.5 = 0.5 (inflection point)
        let input = pixel(0.5, 0.5, 0.5, 1.0);
        let f = Blend { mode: 3, opacity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn blend_vivid_light() {
        // vivid_light at b=0.8 (>0.5): b / (2*(1-b)+eps) = 0.8 / 0.4 = 2.0 → clamped to 1.0
        let input = pixel(0.8, 0.8, 0.8, 1.0);
        let f = Blend { mode: 14, opacity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-4, "vivid_light(0.8) should be ~1.0: {}", out[0]);
    }

    #[test]
    fn blend_vivid_light_dark() {
        // vivid_light at b=0.3 (<=0.5): 1 - (1-0.3)/(2*0.3) = 1 - 0.7/0.6 → clamp(0) = 0
        let input = pixel(0.3, 0.3, 0.3, 1.0);
        let f = Blend { mode: 14, opacity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] <= 0.01, "vivid_light(0.3) should be ~0: {}", out[0]);
    }

    #[test]
    fn blend_linear_light() {
        // linear_light self-blend: 3*b - 1
        // b=0.8 => 2.4-1 = 1.4 → clamped to 1.0
        let input = pixel(0.8, 0.8, 0.8, 1.0);
        let f = Blend { mode: 15, opacity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5, "linear_light(0.8) = {}", out[0]);
        // b=0.5 => 1.5-1 = 0.5
        let input2 = pixel(0.5, 0.5, 0.5, 1.0);
        let out2 = f.compute(&input2, 1, 1).unwrap();
        assert!((out2[0] - 0.5).abs() < 1e-5, "linear_light(0.5) = {}", out2[0]);
    }

    #[test]
    fn blend_pin_light_self_blend_identity() {
        // pin_light with self-blend is always identity
        let input = pixel(0.3, 0.7, 0.5, 1.0);
        let f = Blend { mode: 16, opacity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        for i in 0..3 {
            assert!((out[i] - input[i]).abs() < 1e-5, "pin_light ch{i}: {}", out[i]);
        }
    }

    #[test]
    fn blend_hard_mix_threshold() {
        // hard_mix: 2*b >= 1 → 1, else 0
        let bright = pixel(0.6, 0.6, 0.6, 1.0);
        let f = Blend { mode: 17, opacity: 1.0 };
        let out = f.compute(&bright, 1, 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5, "hard_mix(0.6) should be 1.0");

        let dark = pixel(0.4, 0.4, 0.4, 1.0);
        let out2 = f.compute(&dark, 1, 1).unwrap();
        assert!(out2[0].abs() < 1e-5, "hard_mix(0.4) should be 0.0");
    }

    #[test]
    fn blend_dissolve_does_not_panic() {
        let input = pixel(0.5, 0.5, 0.5, 1.0);
        let f = Blend { mode: 18, opacity: 0.5 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn blend_darker_lighter_color_self_blend_identity() {
        let input = pixel(0.3, 0.7, 0.5, 1.0);
        for mode in [19, 20] {
            let f = Blend { mode, opacity: 1.0 };
            let out = f.compute(&input, 1, 1).unwrap();
            for i in 0..4 {
                assert!((out[i] - input[i]).abs() < 1e-6, "mode {mode} ch{i}");
            }
        }
    }

    #[test]
    fn blend_hsl_modes_self_blend_identity() {
        let input = pixel(0.3, 0.6, 0.9, 1.0);
        for mode in 21..=24 {
            let f = Blend { mode, opacity: 1.0 };
            let out = f.compute(&input, 1, 1).unwrap();
            for i in 0..3 {
                assert!(
                    (out[i] - input[i]).abs() < 0.02,
                    "HSL mode {mode} ch{i}: expected ~{}, got {}",
                    input[i], out[i]
                );
            }
        }
    }

    #[test]
    fn blend_all_modes_no_panic() {
        let input = pixel(0.5, 0.3, 0.8, 1.0);
        for mode in 0..=24 {
            let f = Blend { mode, opacity: 0.7 };
            let out = f.compute(&input, 1, 1).unwrap();
            assert_eq!(out.len(), 4, "mode {mode} output length");
            for i in 0..3 {
                assert!(out[i].is_finite(), "mode {mode} ch{i} is NaN/Inf");
            }
        }
    }

    #[test]
    fn blend_if_shadows_fade() {
        let dark = pixel(0.1, 0.1, 0.1, 1.0);
        let f = BlendIf {
            shadow_threshold: 0.2, shadow_feather: 0.1,
            highlight_threshold: 1.0, highlight_feather: 0.1,
        };
        let out = f.compute(&dark, 1, 1).unwrap();
        assert!(out[3] < 1.0, "dark pixel should have reduced alpha: {}", out[3]);
    }

    #[test]
    fn blend_if_midtone_preserved() {
        let mid = pixel(0.5, 0.5, 0.5, 1.0);
        let f = BlendIf {
            shadow_threshold: 0.0, shadow_feather: 0.1,
            highlight_threshold: 1.0, highlight_feather: 0.1,
        };
        let out = f.compute(&mid, 1, 1).unwrap();
        assert!((out[3] - 1.0).abs() < 1e-6, "midtone should preserve alpha: {}", out[3]);
    }

    #[test]
    fn filters_registered() {
        let names: Vec<&str> = crate::registered_operations()
            .into_iter()
            .filter(|op| op.category == "composite")
            .map(|op| op.name)
            .collect();
        assert!(names.contains(&"premultiply"));
        assert!(names.contains(&"unpremultiply"));
        assert!(names.contains(&"blend"));
        assert!(names.contains(&"blend_if"));
    }
}
