use crate::node::{NodeInfo, PipelineError};
use crate::ops::Filter;

use super::luma;

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

impl Filter for BlendIf {
    fn compute(&self, input: &[f32], w: u32, h: u32) -> Result<Vec<f32>, PipelineError> {
        let info = NodeInfo { width: w, height: h, color_space: crate::color_space::ColorSpace::Linear };
        self.compute_with_info(input, &info)
    }

    fn compute_with_info(&self, input: &[f32], info: &NodeInfo) -> Result<Vec<f32>, PipelineError> {
        let coeffs = info.color_space.luma_coefficients();
        let mut out = input.to_vec();
        let s_lo = self.shadow_threshold;
        let s_hi = s_lo + self.shadow_feather.max(1e-6);
        let h_hi = self.highlight_threshold;
        let h_lo = h_hi - self.highlight_feather.max(1e-6);

        for px in out.chunks_exact_mut(4) {
            let l = luma(px[0], px[1], px[2], coeffs);
            let shadow_mix = ((l - s_lo) / (s_hi - s_lo)).clamp(0.0, 1.0);
            let highlight_mix = ((h_hi - l) / (h_hi - h_lo)).clamp(0.0, 1.0);
            let mask = shadow_mix * highlight_mix;
            px[3] *= mask;
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(BLEND_IF_WGSL)
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let info = NodeInfo { width, height, color_space: crate::color_space::ColorSpace::Linear };
        self.gpu_params_with_info(&info)
    }

    fn gpu_params_with_info(&self, info: &NodeInfo) -> Option<Vec<u8>> {
        let coeffs = info.color_space.luma_coefficients();
        let mut buf = Vec::with_capacity(36);
        buf.extend_from_slice(&info.width.to_le_bytes());
        buf.extend_from_slice(&info.height.to_le_bytes());
        buf.extend_from_slice(&self.shadow_threshold.to_le_bytes());
        buf.extend_from_slice(&self.shadow_feather.max(1e-6).to_le_bytes());
        buf.extend_from_slice(&self.highlight_threshold.to_le_bytes());
        buf.extend_from_slice(&self.highlight_feather.max(1e-6).to_le_bytes());
        buf.extend_from_slice(&coeffs[0].to_le_bytes());
        buf.extend_from_slice(&coeffs[1].to_le_bytes());
        buf.extend_from_slice(&coeffs[2].to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // padding
        Some(buf)
    }
}

const BLEND_IF_WGSL: &str = r#"
struct Params {
    width: u32, height: u32,
    shadow_threshold: f32, shadow_feather: f32,
    highlight_threshold: f32, highlight_feather: f32,
    luma_r: f32, luma_g: f32, luma_b: f32, _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    let luma = params.luma_r * p.x + params.luma_g * p.y + params.luma_b * p.z;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::ColorSpace;
    use crate::node::NodeInfo;

    fn pixel(r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
        vec![r, g, b, a]
    }

    #[test]
    fn blend_if_uses_colorspace_luma() {
        // A pixel that's "dark" in Rec.709 but may differ in AP1
        let input = pixel(0.1, 0.1, 0.1, 1.0);
        let f = BlendIf {
            shadow_threshold: 0.2, shadow_feather: 0.1,
            highlight_threshold: 1.0, highlight_feather: 0.1,
        };

        let info_709 = NodeInfo { width: 1, height: 1, color_space: ColorSpace::Linear };
        let out_709 = f.compute_with_info(&input, &info_709).unwrap();

        let info_ap1 = NodeInfo { width: 1, height: 1, color_space: ColorSpace::AcesCg };
        let out_ap1 = f.compute_with_info(&input, &info_ap1).unwrap();

        // Both should fade the dark pixel (luma ~0.1 < shadow_threshold 0.2)
        assert!(out_709[3] < 1.0, "709: dark pixel should fade");
        assert!(out_ap1[3] < 1.0, "AP1: dark pixel should fade");
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
}
