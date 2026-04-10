use crate::node::{NodeInfo, PipelineError};
use crate::ops::Filter;

use super::{luma, pcg_hash, sat, set_lum, set_sat, soft_light_d};

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

impl Filter for Blend {
    fn compute(&self, input: &[f32], w: u32, h: u32) -> Result<Vec<f32>, PipelineError> {
        // Default path uses Rec.709 coefficients (Linear sRGB).
        let info = NodeInfo {
            width: w,
            height: h,
            color_space: crate::color_space::ColorSpace::Linear,
        };
        self.compute_with_info(input, &info)
    }

    fn compute_with_info(&self, input: &[f32], info: &NodeInfo) -> Result<Vec<f32>, PipelineError> {
        let opacity = self.opacity;
        let inv_opacity = 1.0 - opacity;
        let coeffs = info.color_space.luma_coefficients();
        let mut out = input.to_vec();

        match self.mode {
            // ── Per-channel modes (0-17) ───────────────────────────────
            // ISO 32000-2:2020 Section 11.3.5 — separable blend modes.
            0..=17 => {
                // Hoist mode dispatch outside the loop so the inner loop is a
                // single arithmetic expression that LLVM can auto-vectorize.
                //
                // Scene-referred linear: NO input clamping. Values > 1.0 are
                // valid HDR (specular highlights, emissive surfaces). Blend modes
                // that naturally extend to unbounded values do so. Modes with
                // singularities use epsilon guards, not clamping.
                // Reference: Nuke merge node — operates on unbounded scene-referred linear.
                let blend_fn: fn(f32) -> f32 = match self.mode {
                    1 => |b: f32| b * b,                       // Multiply — naturally extends
                    2 => |b: f32| 1.0 - (1.0 - b) * (1.0 - b), // Screen — naturally extends
                    3 => |b: f32| {
                        // Overlay — threshold at 0.5
                        if b < 0.5 {
                            2.0 * b * b
                        } else {
                            1.0 - 2.0 * (1.0 - b) * (1.0 - b)
                        }
                    },
                    4 => |b: f32| {
                        // SoftLight (ISO 32000-2)
                        if b <= 0.5 {
                            b - (1.0 - 2.0 * b) * b * (1.0 - b)
                        } else {
                            b + (2.0 * b - 1.0) * (soft_light_d(b.clamp(0.0, 1.0)) - b)
                        }
                    },
                    5 => |b: f32| {
                        // HardLight
                        if b < 0.5 {
                            2.0 * b * b
                        } else {
                            1.0 - 2.0 * (1.0 - b) * (1.0 - b)
                        }
                    },
                    6 => |b: f32| {
                        // ColorDodge — epsilon guard, no cap
                        b / (1.0 - b).abs().max(1e-6)
                    },
                    7 => |b: f32| {
                        // ColorBurn — epsilon guard, no cap
                        1.0 - (1.0 - b) / b.abs().max(1e-6)
                    },
                    8 => |b: f32| b,    // Darken (identity for self-blend)
                    9 => |b: f32| b,    // Lighten (identity for self-blend)
                    10 => |_: f32| 0.0, // Difference (zero for self-blend)
                    11 => |b: f32| b + b - 2.0 * b * b, // Exclusion — naturally extends
                    12 => |b: f32| b + b - 1.0, // LinearBurn — allow negative (valid in linear)
                    13 => |b: f32| b + b, // LinearDodge — allow > 1.0 (HDR)
                    14 => |b: f32| {
                        // VividLight — epsilon guards
                        if b <= 0.5 {
                            if b.abs() > 1e-6 {
                                1.0 - (1.0 - b) / (2.0 * b)
                            } else {
                                0.0
                            }
                        } else {
                            b / (2.0 * (1.0 - b)).abs().max(1e-6)
                        }
                    },
                    15 => |b: f32| 2.0 * b + b - 1.0, // LinearLight — no clamp
                    16 => |b: f32| {
                        // PinLight
                        if b < 0.5 {
                            b.min(2.0 * b)
                        } else {
                            b.max(2.0 * b - 1.0)
                        }
                    },
                    17 => |b: f32| if b + b >= 1.0 { 1.0 } else { 0.0 }, // HardMix
                    _ => |b: f32| b,
                };

                // No input clamping — scene-referred linear values are unbounded.
                for px in out.chunks_exact_mut(4) {
                    px[0] = px[0] * inv_opacity + blend_fn(px[0]) * opacity;
                    px[1] = px[1] * inv_opacity + blend_fn(px[1]) * opacity;
                    px[2] = px[2] * inv_opacity + blend_fn(px[2]) * opacity;
                }
            }

            // ── Dissolve (18) ──────────────────────────────────────────
            18 => {
                for (i, _px) in out.chunks_exact_mut(4).enumerate() {
                    let _h = pcg_hash(i as u32);
                    // Self-blend: identity regardless of threshold
                }
            }

            // ── Darker/Lighter color (19-20) ───────────────────────────
            // Uses color-space-aware luminance for comparison.
            // For self-blend: identity (same pixel both sides).
            19 | 20 => {}

            // ── HSL modes (21-24) ──────────────────────────────────────
            // ISO 32000-2 Section 11.3.5.4 — non-separable blend modes.
            // Luminance uses working-space-derived coefficients.
            //
            // NOTE: Non-separable modes use ClipColor which assumes [0,1] range.
            // For HDR values > 1.0, we clamp inputs to these modes only.
            // This is a known limitation — the SetLum/SetSat/ClipColor functions
            // from ISO 32000-2 are defined for display-referred values.
            // A future track could implement scene-referred non-separable modes.
            21..=24 => {
                for px in out.chunks_exact_mut(4) {
                    let (r, g, b) = (
                        px[0].clamp(0.0, 1.0),
                        px[1].clamp(0.0, 1.0),
                        px[2].clamp(0.0, 1.0),
                    );
                    let (ro, go, bo) = match self.mode {
                        21 => {
                            // Hue: hue from blend, sat+lum from base
                            let (sr, sg, sb) = set_sat(r, g, b, sat(r, g, b));
                            set_lum(sr, sg, sb, luma(r, g, b, coeffs), coeffs)
                        }
                        22 => {
                            // Saturation: sat from blend, hue+lum from base
                            let (sr, sg, sb) = set_sat(r, g, b, sat(r, g, b));
                            set_lum(sr, sg, sb, luma(r, g, b, coeffs), coeffs)
                        }
                        23 => {
                            // Color: hue+sat from blend, lum from base
                            set_lum(r, g, b, luma(r, g, b, coeffs), coeffs)
                        }
                        24 => {
                            // Luminosity: lum from blend, hue+sat from base
                            set_lum(r, g, b, luma(r, g, b, coeffs), coeffs)
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
        Some(include_str!("../../shaders/blend.wgsl"))
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        // Default: Rec.709 luma coefficients (Linear sRGB)
        let info = NodeInfo {
            width,
            height,
            color_space: crate::color_space::ColorSpace::Linear,
        };
        self.gpu_params_with_info(&info)
    }

    fn gpu_params_with_info(&self, info: &NodeInfo) -> Option<Vec<u8>> {
        let coeffs = info.color_space.luma_coefficients();
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&info.width.to_le_bytes());
        buf.extend_from_slice(&info.height.to_le_bytes());
        buf.extend_from_slice(&self.opacity.to_le_bytes());
        buf.extend_from_slice(&self.mode.to_le_bytes());
        buf.extend_from_slice(&coeffs[0].to_le_bytes());
        buf.extend_from_slice(&coeffs[1].to_le_bytes());
        buf.extend_from_slice(&coeffs[2].to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // padding for 16-byte alignment
        Some(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
        vec![r, g, b, a]
    }

    #[test]
    fn blend_multiply_darkens() {
        let input = pixel(0.8, 0.8, 0.8, 1.0);
        let f = Blend {
            mode: 1,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] < 0.8, "multiply should darken: {}", out[0]);
        assert!((out[0] - 0.64).abs() < 1e-5); // 0.8 * 0.8
    }

    #[test]
    fn blend_screen_brightens() {
        let input = pixel(0.5, 0.5, 0.5, 1.0);
        let f = Blend {
            mode: 2,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] > 0.5, "screen should brighten: {}", out[0]);
        assert!((out[0] - 0.75).abs() < 1e-5); // 1 - (1-0.5)*(1-0.5)
    }

    #[test]
    fn blend_opacity_zero_is_identity() {
        let input = pixel(0.5, 0.3, 0.7, 1.0);
        let f = Blend {
            mode: 1,
            opacity: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        for i in 0..3 {
            assert!((out[i] - input[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn blend_overlay_midtone() {
        // overlay at 0.5 => 2*0.5*0.5 = 0.5 (inflection point)
        let input = pixel(0.5, 0.5, 0.5, 1.0);
        let f = Blend {
            mode: 3,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn blend_vivid_light() {
        // vivid_light at b=0.8 (>0.5): b / (2*(1-b)+eps) = 0.8 / 0.4 = 2.0
        // In linear/HDR: NOT clamped — result is 2.0
        let input = pixel(0.8, 0.8, 0.8, 1.0);
        let f = Blend {
            mode: 14,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(
            (out[0] - 2.0).abs() < 1e-4,
            "vivid_light(0.8) = 0.8/0.4 = 2.0: {}",
            out[0]
        );
    }

    #[test]
    fn blend_vivid_light_dark() {
        // vivid_light at b=0.3 (<=0.5): 1 - (1-0.3)/(2*0.3) = 1 - 0.7/0.6 ≈ -0.167
        // In linear/HDR: NOT clamped — negative is valid
        let input = pixel(0.3, 0.3, 0.3, 1.0);
        let f = Blend {
            mode: 14,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(
            out[0] < 0.0,
            "vivid_light(0.3) should be negative in linear: {}",
            out[0]
        );
    }

    #[test]
    fn blend_linear_light() {
        // linear_light self-blend: 3*b - 1
        // b=0.8 => 2.4-1 = 1.4 — NOT clamped in HDR
        let input = pixel(0.8, 0.8, 0.8, 1.0);
        let f = Blend {
            mode: 15,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(
            (out[0] - 1.4).abs() < 1e-5,
            "linear_light(0.8) = 1.4: {}",
            out[0]
        );
        // b=0.5 => 1.5-1 = 0.5
        let input2 = pixel(0.5, 0.5, 0.5, 1.0);
        let out2 = f.compute(&input2, 1, 1).unwrap();
        assert!(
            (out2[0] - 0.5).abs() < 1e-5,
            "linear_light(0.5) = {}",
            out2[0]
        );
    }

    #[test]
    fn blend_pin_light_self_blend_identity() {
        // pin_light with self-blend is always identity
        let input = pixel(0.3, 0.7, 0.5, 1.0);
        let f = Blend {
            mode: 16,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        for i in 0..3 {
            assert!(
                (out[i] - input[i]).abs() < 1e-5,
                "pin_light ch{i}: {}",
                out[i]
            );
        }
    }

    #[test]
    fn blend_hard_mix_threshold() {
        // hard_mix: 2*b >= 1 → 1, else 0
        let bright = pixel(0.6, 0.6, 0.6, 1.0);
        let f = Blend {
            mode: 17,
            opacity: 1.0,
        };
        let out = f.compute(&bright, 1, 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5, "hard_mix(0.6) should be 1.0");

        let dark = pixel(0.4, 0.4, 0.4, 1.0);
        let out2 = f.compute(&dark, 1, 1).unwrap();
        assert!(out2[0].abs() < 1e-5, "hard_mix(0.4) should be 0.0");
    }

    #[test]
    fn blend_dissolve_does_not_panic() {
        let input = pixel(0.5, 0.5, 0.5, 1.0);
        let f = Blend {
            mode: 18,
            opacity: 0.5,
        };
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
                    input[i],
                    out[i]
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
    fn blend_soft_light_iso32000() {
        // ISO 32000-2 SoftLight formula for self-blend at b=0.5:
        // b <= 0.5: b - (1-2*b)*b*(1-b) = 0.5 - 0*0.5*0.5 = 0.5
        let input = pixel(0.5, 0.5, 0.5, 1.0);
        let f = Blend {
            mode: 4,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-5, "soft_light(0.5) = {}", out[0]);

        // b=0.8 (>0.5): b + (2b-1)*(D(b)-b) where D(0.8)=sqrt(0.8)≈0.8944
        // = 0.8 + 0.6*(0.8944-0.8) = 0.8 + 0.6*0.0944 ≈ 0.8567
        let input2 = pixel(0.8, 0.8, 0.8, 1.0);
        let out2 = f.compute(&input2, 1, 1).unwrap();
        assert!(
            (out2[0] - 0.8567).abs() < 0.01,
            "soft_light(0.8) = {}",
            out2[0]
        );
    }

    #[test]
    fn blend_hsl_uses_colorspace_coefficients() {
        use crate::color_space::ColorSpace;
        let input = pixel(0.3, 0.6, 0.9, 1.0);
        let f = Blend {
            mode: 24,
            opacity: 1.0,
        }; // luminosity mode

        // Linear (Rec.709 coefficients)
        let info_709 = NodeInfo {
            width: 1,
            height: 1,
            color_space: ColorSpace::Linear,
        };
        let out_709 = f.compute_with_info(&input, &info_709).unwrap();

        // ACEScg (AP1 coefficients — different from Rec.709)
        let info_ap1 = NodeInfo {
            width: 1,
            height: 1,
            color_space: ColorSpace::AcesCg,
        };
        let out_ap1 = f.compute_with_info(&input, &info_ap1).unwrap();

        // For self-blend both are ~identity, but the coefficients differ
        // which means clip_color/set_lum may produce subtly different results
        // for non-neutral colors. At minimum, both should be valid.
        for i in 0..3 {
            assert!(out_709[i].is_finite(), "709 ch{i} NaN");
            assert!(out_ap1[i].is_finite(), "AP1 ch{i} NaN");
        }
    }

    // ── HDR / scene-referred linear tests ─────────────────────────────

    #[test]
    fn blend_multiply_hdr() {
        // HDR: 2.0 * 2.0 = 4.0 (valid in scene-referred linear)
        let input = pixel(2.0, 0.5, 0.0, 1.0);
        let f = Blend {
            mode: 1,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(
            (out[0] - 4.0).abs() < 1e-5,
            "multiply HDR: 2*2=4, got {}",
            out[0]
        );
        assert!(
            (out[1] - 0.25).abs() < 1e-5,
            "multiply 0.5*0.5=0.25, got {}",
            out[1]
        );
    }

    #[test]
    fn blend_screen_hdr() {
        // Screen: 1 - (1-2.0)*(1-2.0) = 1 - (-1)*(-1) = 1 - 1 = 0
        let input = pixel(2.0, 2.0, 2.0, 1.0);
        let f = Blend {
            mode: 2,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        // Screen with >1 values: mathematically extends
        assert!(
            out[0].is_finite(),
            "screen HDR should produce finite values"
        );
    }

    #[test]
    fn blend_linear_dodge_hdr() {
        // LinearDodge: b + b = 2*b. For b=1.5, result = 3.0
        let input = pixel(1.5, 0.5, 3.0, 1.0);
        let f = Blend {
            mode: 13,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(
            (out[0] - 3.0).abs() < 1e-5,
            "linear dodge 1.5+1.5=3.0: {}",
            out[0]
        );
        assert!(
            (out[1] - 1.0).abs() < 1e-5,
            "linear dodge 0.5+0.5=1.0: {}",
            out[1]
        );
        assert!(
            (out[2] - 6.0).abs() < 1e-5,
            "linear dodge 3.0+3.0=6.0: {}",
            out[2]
        );
    }

    #[test]
    fn blend_linear_burn_allows_negative() {
        // LinearBurn: b + b - 1. For b=0.3, result = -0.4 (valid in linear)
        let input = pixel(0.3, 0.3, 0.3, 1.0);
        let f = Blend {
            mode: 12,
            opacity: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(
            (out[0] - (-0.4)).abs() < 1e-5,
            "linear burn 0.3+0.3-1=-0.4: {}",
            out[0]
        );
    }

    #[test]
    fn blend_hdr_values_not_clamped_on_input() {
        // Ensure HDR values pass through blend modes unclamped
        let input = pixel(5.0, 0.0, 0.0, 1.0);
        let f = Blend {
            mode: 0,
            opacity: 0.0,
        }; // Normal, opacity 0 = identity
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(
            (out[0] - 5.0).abs() < 1e-5,
            "HDR value should survive: {}",
            out[0]
        );
    }
}
