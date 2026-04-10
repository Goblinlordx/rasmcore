//! Dual-input blend — blends foreground over background using standard blend modes.
//!
//! This is the proper two-image version of the blend filter. The single-input
//! `Blend` filter self-blends (input against itself); this compositor takes
//! two separate images and blends them with the full set of 25 Photoshop/ISO modes.
//!
//! Formula: `output = mix(bg, blend(fg, bg), opacity)`
//!
//! Reference: ISO 32000-2:2020 Section 11.3.5 — PDF 2.0 Blend Modes.

use crate::compositor_node::CompositorNode;
use crate::node::{NodeInfo, PipelineError};
use crate::ops::Compositor;
use crate::registry::{CompositorFactoryRegistration, ParamDescriptor, ParamType};

// ─── Static registration ────────────────────────────────────────────────────

inventory::submit! {
    &BlendDual::REGISTRATION as &'static CompositorFactoryRegistration
}

impl BlendDual {
    const PARAMS: &[ParamDescriptor] = &[
        ParamDescriptor {
            name: "mode",
            value_type: ParamType::U32,
            min: Some(0.0),
            max: Some(24.0),
            step: Some(1.0),
            default: Some(1.0),
            hint: Some("rc.enum"),
            description: "Blend mode (0=normal..24=luminosity)",
            constraints: &[],
        },
        ParamDescriptor {
            name: "opacity",
            value_type: ParamType::F32,
            min: Some(0.0),
            max: Some(1.0),
            step: Some(0.05),
            default: Some(1.0),
            hint: None,
            description: "Blend opacity (0=bg only, 1=fully blended)",
            constraints: &[],
        },
    ];

    pub const REGISTRATION: CompositorFactoryRegistration = CompositorFactoryRegistration {
        name: "blend_dual",
        display_name: "Blend (Dual Input)",
        category: "composite",
        params: Self::PARAMS,
        cost: "O(n)",
        factory: |upstream_a, upstream_b, info, params| {
            let mode = params.ints.get("mode").copied().unwrap_or(1) as u32;
            let opacity = params.floats.get("opacity").copied().unwrap_or(1.0);
            Box::new(CompositorNode::new(
                upstream_a,
                upstream_b,
                info,
                BlendDual { mode, opacity },
            ))
        },
    };
}

use super::{luma, pcg_hash, sat, set_lum, set_sat, soft_light_d};

/// Dual-input blend compositor — blends fg over bg using standard blend modes.
///
/// Modes:
///   0=normal, 1=multiply, 2=screen, 3=overlay, 4=soft_light,
///   5=hard_light, 6=color_dodge, 7=color_burn, 8=darken, 9=lighten,
///   10=difference, 11=exclusion, 12=linear_burn, 13=linear_dodge,
///   14=vivid_light, 15=linear_light, 16=pin_light, 17=hard_mix,
///   18=dissolve, 19=darker_color, 20=lighter_color,
///   21=hue, 22=saturation, 23=color, 24=luminosity
#[derive(Clone)]
pub struct BlendDual {
    /// Blend mode (0-24)
    pub mode: u32,
    /// Blend opacity (0 = bg only, 1 = fully blended)
    pub opacity: f32,
}

impl BlendDual {
    fn gpu_params_data(&self, info: &NodeInfo) -> Vec<u8> {
        let coeffs = info.color_space.luma_coefficients();
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&info.width.to_le_bytes());
        buf.extend_from_slice(&info.height.to_le_bytes());
        buf.extend_from_slice(&self.opacity.to_le_bytes());
        buf.extend_from_slice(&self.mode.to_le_bytes());
        buf.extend_from_slice(&coeffs[0].to_le_bytes());
        buf.extend_from_slice(&coeffs[1].to_le_bytes());
        buf.extend_from_slice(&coeffs[2].to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // padding
        buf
    }
}

impl Compositor for BlendDual {
    fn compute(
        &self,
        fg: &[f32],
        bg: &[f32],
        w: u32,
        h: u32,
    ) -> Result<Vec<f32>, PipelineError> {
        let info = NodeInfo {
            width: w,
            height: h,
            color_space: crate::color_space::ColorSpace::Linear,
        };
        self.compute_with_info(fg, bg, &info)
    }

    fn compute_with_info(
        &self,
        fg: &[f32],
        bg: &[f32],
        info: &NodeInfo,
    ) -> Result<Vec<f32>, PipelineError> {
        let opacity = self.opacity;
        let inv_opacity = 1.0 - opacity;
        let coeffs = info.color_space.luma_coefficients();
        let mut out = Vec::with_capacity(bg.len());

        match self.mode {
            // ── Per-channel modes (0-17) ───────────────────────────────
            0..=17 => {
                let blend_fn: fn(f32, f32) -> f32 = match self.mode {
                    0 => |a: f32, _b: f32| a, // Normal — fg replaces bg
                    1 => |a: f32, b: f32| a * b, // Multiply
                    2 => |a: f32, b: f32| 1.0 - (1.0 - a) * (1.0 - b), // Screen
                    3 => |a: f32, b: f32| {
                        // Overlay — uses bg (base) as switch
                        if b < 0.5 {
                            2.0 * a * b
                        } else {
                            1.0 - 2.0 * (1.0 - a) * (1.0 - b)
                        }
                    },
                    4 => |a: f32, b: f32| {
                        // SoftLight (ISO 32000-2) — fg modifies bg
                        if a <= 0.5 {
                            b - (1.0 - 2.0 * a) * b * (1.0 - b)
                        } else {
                            b + (2.0 * a - 1.0) * (soft_light_d(b.clamp(0.0, 1.0)) - b)
                        }
                    },
                    5 => |a: f32, b: f32| {
                        // HardLight — uses fg (blend) as switch
                        if a < 0.5 {
                            2.0 * a * b
                        } else {
                            1.0 - 2.0 * (1.0 - a) * (1.0 - b)
                        }
                    },
                    6 => |a: f32, b: f32| {
                        // ColorDodge
                        b / (1.0 - a).abs().max(1e-6)
                    },
                    7 => |a: f32, b: f32| {
                        // ColorBurn
                        1.0 - (1.0 - b) / a.abs().max(1e-6)
                    },
                    8 => |a: f32, b: f32| a.min(b), // Darken
                    9 => |a: f32, b: f32| a.max(b), // Lighten
                    10 => |a: f32, b: f32| (a - b).abs(), // Difference
                    11 => |a: f32, b: f32| a + b - 2.0 * a * b, // Exclusion
                    12 => |a: f32, b: f32| a + b - 1.0, // LinearBurn
                    13 => |a: f32, b: f32| a + b, // LinearDodge
                    14 => |a: f32, b: f32| {
                        // VividLight
                        if a <= 0.5 {
                            if a.abs() > 1e-6 {
                                1.0 - (1.0 - b) / (2.0 * a)
                            } else {
                                0.0
                            }
                        } else {
                            b / (2.0 * (1.0 - a)).abs().max(1e-6)
                        }
                    },
                    15 => |a: f32, b: f32| 2.0 * a + b - 1.0, // LinearLight
                    16 => |a: f32, b: f32| {
                        // PinLight
                        if a < 0.5 {
                            b.min(2.0 * a)
                        } else {
                            b.max(2.0 * a - 1.0)
                        }
                    },
                    17 => |a: f32, b: f32| if a + b >= 1.0 { 1.0 } else { 0.0 }, // HardMix
                    _ => |_a: f32, b: f32| b,
                };

                for (fg_px, bg_px) in fg.chunks_exact(4).zip(bg.chunks_exact(4)) {
                    out.push(bg_px[0] * inv_opacity + blend_fn(fg_px[0], bg_px[0]) * opacity);
                    out.push(bg_px[1] * inv_opacity + blend_fn(fg_px[1], bg_px[1]) * opacity);
                    out.push(bg_px[2] * inv_opacity + blend_fn(fg_px[2], bg_px[2]) * opacity);
                    out.push(bg_px[3]); // alpha from background
                }
            }

            // ── Dissolve (18) ──────────────────────────────────────────
            18 => {
                for (i, (fg_px, bg_px)) in
                    fg.chunks_exact(4).zip(bg.chunks_exact(4)).enumerate()
                {
                    let h = pcg_hash(i as u32);
                    let threshold = (h & 0xFF) as f32 / 255.0;
                    if threshold < opacity {
                        out.extend_from_slice(fg_px);
                    } else {
                        out.extend_from_slice(bg_px);
                    }
                }
            }

            // ── Darker/Lighter color (19-20) ───────────────────────────
            19 => {
                for (fg_px, bg_px) in fg.chunks_exact(4).zip(bg.chunks_exact(4)) {
                    let fg_l = luma(fg_px[0], fg_px[1], fg_px[2], coeffs);
                    let bg_l = luma(bg_px[0], bg_px[1], bg_px[2], coeffs);
                    let (r, g, b) = if fg_l < bg_l {
                        (fg_px[0], fg_px[1], fg_px[2])
                    } else {
                        (bg_px[0], bg_px[1], bg_px[2])
                    };
                    out.push(bg_px[0] * inv_opacity + r * opacity);
                    out.push(bg_px[1] * inv_opacity + g * opacity);
                    out.push(bg_px[2] * inv_opacity + b * opacity);
                    out.push(bg_px[3]);
                }
            }
            20 => {
                for (fg_px, bg_px) in fg.chunks_exact(4).zip(bg.chunks_exact(4)) {
                    let fg_l = luma(fg_px[0], fg_px[1], fg_px[2], coeffs);
                    let bg_l = luma(bg_px[0], bg_px[1], bg_px[2], coeffs);
                    let (r, g, b) = if fg_l > bg_l {
                        (fg_px[0], fg_px[1], fg_px[2])
                    } else {
                        (bg_px[0], bg_px[1], bg_px[2])
                    };
                    out.push(bg_px[0] * inv_opacity + r * opacity);
                    out.push(bg_px[1] * inv_opacity + g * opacity);
                    out.push(bg_px[2] * inv_opacity + b * opacity);
                    out.push(bg_px[3]);
                }
            }

            // ── HSL modes (21-24) ──────────────────────────────────────
            21..=24 => {
                for (fg_px, bg_px) in fg.chunks_exact(4).zip(bg.chunks_exact(4)) {
                    let (ar, ag, ab) = (
                        fg_px[0].clamp(0.0, 1.0),
                        fg_px[1].clamp(0.0, 1.0),
                        fg_px[2].clamp(0.0, 1.0),
                    );
                    let (br, bg_r, bb) = (
                        bg_px[0].clamp(0.0, 1.0),
                        bg_px[1].clamp(0.0, 1.0),
                        bg_px[2].clamp(0.0, 1.0),
                    );
                    let (ro, go, bo) = match self.mode {
                        21 => {
                            // Hue: hue from fg, sat+lum from bg
                            let (sr, sg, sb) = set_sat(ar, ag, ab, sat(br, bg_r, bb));
                            set_lum(sr, sg, sb, luma(br, bg_r, bb, coeffs), coeffs)
                        }
                        22 => {
                            // Saturation: sat from fg, hue+lum from bg
                            let (sr, sg, sb) = set_sat(br, bg_r, bb, sat(ar, ag, ab));
                            set_lum(sr, sg, sb, luma(br, bg_r, bb, coeffs), coeffs)
                        }
                        23 => {
                            // Color: hue+sat from fg, lum from bg
                            set_lum(ar, ag, ab, luma(br, bg_r, bb, coeffs), coeffs)
                        }
                        24 => {
                            // Luminosity: lum from fg, hue+sat from bg
                            set_lum(br, bg_r, bb, luma(ar, ag, ab, coeffs), coeffs)
                        }
                        _ => (br, bg_r, bb),
                    };
                    out.push(bg_px[0] * inv_opacity + ro * opacity);
                    out.push(bg_px[1] * inv_opacity + go * opacity);
                    out.push(bg_px[2] * inv_opacity + bo * opacity);
                    out.push(bg_px[3]);
                }
            }

            _ => {
                // Unknown mode: return bg unchanged
                out.extend_from_slice(bg);
            }
        }

        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(include_str!("../../shaders/blend_dual.wgsl"))
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let info = NodeInfo {
            width,
            height,
            color_space: crate::color_space::ColorSpace::Linear,
        };
        Some(self.gpu_params_data(&info))
    }

    fn gpu_params_with_info(&self, info: &NodeInfo) -> Option<Vec<u8>> {
        Some(self.gpu_params_data(info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
        vec![r, g, b, a]
    }

    #[test]
    fn normal_mode_replaces_with_fg() {
        let fg = px(1.0, 0.0, 0.0, 1.0);
        let bg = px(0.0, 0.0, 1.0, 1.0);
        let comp = BlendDual {
            mode: 0,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!(out[2].abs() < 1e-5);
    }

    #[test]
    fn multiply_mode() {
        let fg = px(0.5, 0.8, 0.2, 1.0);
        let bg = px(0.4, 0.6, 1.0, 1.0);
        let comp = BlendDual {
            mode: 1,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!((out[0] - 0.2).abs() < 1e-5, "R: 0.5*0.4=0.2, got {}", out[0]);
        assert!((out[1] - 0.48).abs() < 1e-5, "G: 0.8*0.6=0.48, got {}", out[1]);
        assert!((out[2] - 0.2).abs() < 1e-5, "B: 0.2*1.0=0.2, got {}", out[2]);
    }

    #[test]
    fn screen_mode() {
        let fg = px(0.5, 0.5, 0.5, 1.0);
        let bg = px(0.5, 0.5, 0.5, 1.0);
        let comp = BlendDual {
            mode: 2,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        // 1-(1-0.5)*(1-0.5) = 0.75
        assert!((out[0] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn difference_mode() {
        let fg = px(0.8, 0.3, 0.5, 1.0);
        let bg = px(0.3, 0.8, 0.5, 1.0);
        let comp = BlendDual {
            mode: 10,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
        assert!(out[2].abs() < 1e-5);
    }

    #[test]
    fn opacity_zero_returns_bg() {
        let fg = px(1.0, 0.0, 0.0, 1.0);
        let bg = px(0.0, 1.0, 0.0, 1.0);
        let comp = BlendDual {
            mode: 0,
            opacity: 0.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn darken_mode() {
        let fg = px(0.3, 0.9, 0.5, 1.0);
        let bg = px(0.7, 0.1, 0.5, 1.0);
        let comp = BlendDual {
            mode: 8,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!((out[0] - 0.3).abs() < 1e-5);
        assert!((out[1] - 0.1).abs() < 1e-5);
        assert!((out[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn lighten_mode() {
        let fg = px(0.3, 0.9, 0.5, 1.0);
        let bg = px(0.7, 0.1, 0.5, 1.0);
        let comp = BlendDual {
            mode: 9,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!((out[0] - 0.7).abs() < 1e-5);
        assert!((out[1] - 0.9).abs() < 1e-5);
        assert!((out[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn all_modes_no_panic() {
        let fg = px(0.5, 0.3, 0.8, 1.0);
        let bg = px(0.2, 0.7, 0.4, 1.0);
        for mode in 0..=24 {
            let comp = BlendDual {
                mode,
                opacity: 0.7,
            };
            let out = comp.compute(&fg, &bg, 1, 1).unwrap();
            assert_eq!(out.len(), 4, "mode {mode} output length");
            for i in 0..3 {
                assert!(out[i].is_finite(), "mode {mode} ch{i} is NaN/Inf");
            }
        }
    }

    #[test]
    fn overlay_dark_bg() {
        let fg = px(0.8, 0.8, 0.8, 1.0);
        let bg = px(0.2, 0.2, 0.2, 1.0);
        let comp = BlendDual {
            mode: 3,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        // bg < 0.5: 2*a*b = 2*0.8*0.2 = 0.32
        assert!((out[0] - 0.32).abs() < 1e-5, "overlay dark: {}", out[0]);
    }

    #[test]
    fn overlay_bright_bg() {
        let fg = px(0.8, 0.8, 0.8, 1.0);
        let bg = px(0.8, 0.8, 0.8, 1.0);
        let comp = BlendDual {
            mode: 3,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        // bg >= 0.5: 1-2*(1-a)*(1-b) = 1-2*0.2*0.2 = 0.92
        assert!((out[0] - 0.92).abs() < 1e-5, "overlay bright: {}", out[0]);
    }

    #[test]
    fn linear_dodge_adds() {
        let fg = px(0.3, 0.5, 0.7, 1.0);
        let bg = px(0.2, 0.4, 0.1, 1.0);
        let comp = BlendDual {
            mode: 13,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.9).abs() < 1e-5);
        assert!((out[2] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn hard_mix_threshold() {
        let fg = px(0.6, 0.3, 0.5, 1.0);
        let bg = px(0.5, 0.5, 0.5, 1.0);
        let comp = BlendDual {
            mode: 17,
            opacity: 1.0,
        };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        // fg+bg >= 1: 0.6+0.5=1.1 → 1, 0.3+0.5=0.8 → 0, 0.5+0.5=1.0 → 1
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!(out[1].abs() < 1e-5);
        assert!((out[2] - 1.0).abs() < 1e-5);
    }
}
