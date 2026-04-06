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
            let a = px[3];
            if a > 1e-7 {
                let inv_a = 1.0 / a;
                px[0] *= inv_a;
                px[1] *= inv_a;
                px[2] *= inv_a;
            }
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
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "blend", category = "composite", cost = "O(n)")]
pub struct Blend {
    /// Blend mode: 0=normal, 1=multiply, 2=screen, 3=overlay, 4=soft_light,
    /// 5=hard_light, 6=color_dodge, 7=color_burn, 8=darken, 9=lighten,
    /// 10=difference, 11=exclusion, 12=linear_burn, 13=linear_dodge
    #[param(min = 0, max = 13, step = 1, default = 1)]
    pub mode: u32,
    /// Blend opacity (0 = original, 1 = fully blended)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub opacity: f32,
}

/// Apply blend mode to a single channel.
fn blend_channel(base: f32, mode: u32) -> f32 {
    match mode {
        0 => base,                                                      // normal (identity)
        1 => base * base,                                               // multiply
        2 => 1.0 - (1.0 - base) * (1.0 - base),                       // screen
        3 => if base < 0.5 { 2.0 * base * base } else { 1.0 - 2.0 * (1.0 - base) * (1.0 - base) }, // overlay
        4 => {                                                           // soft light
            if base < 0.5 { base * (base + 0.5) }
            else { 1.0 - (1.0 - base) * (1.5 - base) }
        }
        5 => if base < 0.5 { 2.0 * base * base } else { 1.0 - 2.0 * (1.0 - base) * (1.0 - base) }, // hard light (same as overlay for self-blend)
        6 => if base < 1.0 { (base / (1.0 - base + 1e-6)).min(1.0) } else { 1.0 }, // color dodge
        7 => if base > 0.0 { 1.0 - ((1.0 - base) / (base + 1e-6)).min(1.0) } else { 0.0 }, // color burn
        8 => base.min(base),                                            // darken (identity for self-blend)
        9 => base.max(base),                                            // lighten (identity for self-blend)
        10 => (base - base).abs(),                                       // difference (zero for self-blend)
        11 => base + base - 2.0 * base * base,                          // exclusion
        12 => (base + base - 1.0).max(0.0),                             // linear burn
        13 => (base + base).min(1.0),                                    // linear dodge (add)
        _ => base,
    }
}

impl Filter for Blend {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let mode = self.mode;
        let opacity = self.opacity;
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            let br = blend_channel(px[0].clamp(0.0, 1.0), mode);
            let bg = blend_channel(px[1].clamp(0.0, 1.0), mode);
            let bb = blend_channel(px[2].clamp(0.0, 1.0), mode);
            px[0] = px[0] * (1.0 - opacity) + br * opacity;
            px[1] = px[1] * (1.0 - opacity) + bg * opacity;
            px[2] = px[2] * (1.0 - opacity) + bb * opacity;
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(BLEND_WGSL)
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

const BLEND_WGSL: &str = r#"
struct Params { width: u32, height: u32, opacity: f32, mode: u32, }
@group(0) @binding(2) var<uniform> params: Params;

fn blend_ch(b: f32, mode: u32) -> f32 {
    switch (mode) {
        case 1u:  { return b * b; }
        case 2u:  { return 1.0 - (1.0 - b) * (1.0 - b); }
        case 3u:  { if (b < 0.5) { return 2.0 * b * b; } else { return 1.0 - 2.0 * (1.0 - b) * (1.0 - b); } }
        case 4u:  { if (b < 0.5) { return b * (b + 0.5); } else { return 1.0 - (1.0 - b) * (1.5 - b); } }
        case 5u:  { if (b < 0.5) { return 2.0 * b * b; } else { return 1.0 - 2.0 * (1.0 - b) * (1.0 - b); } }
        case 6u:  { return min(b / (1.0 - b + 0.000001), 1.0); }
        case 7u:  { return max(1.0 - (1.0 - b) / (b + 0.000001), 0.0); }
        case 8u:  { return b; }
        case 9u:  { return b; }
        case 10u: { return 0.0; }
        case 11u: { return b + b - 2.0 * b * b; }
        case 12u: { return max(b + b - 1.0, 0.0); }
        case 13u: { return min(b + b, 1.0); }
        default:  { return b; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    let b = clamp(p, vec4(0.0), vec4(1.0));
    let blended = vec4(blend_ch(b.x, params.mode), blend_ch(b.y, params.mode), blend_ch(b.z, params.mode), b.w);
    let result = mix(p, blended, params.opacity);
    store_pixel(idx, vec4(result.xyz, p.w));
}
"#;

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
