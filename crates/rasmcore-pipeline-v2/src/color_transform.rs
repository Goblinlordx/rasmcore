//! Color transform resource — IDT, LMT, Output Transform, CDL, CSC.
//!
//! Registered once per WASM instance (like Font), referenced by u32 ID
//! in pipeline operations. Supports built-in presets and file-based
//! imports (CLF, .cube, CDL).

use crate::color_math;
use crate::color_space::ColorSpace;
use crate::lmt::Lmt;
use crate::node::{Node, NodeInfo, NodeCapabilities, PipelineError, Upstream, InputRectEstimate};
use crate::rect::Rect;

/// What kind of transform this is (metadata for validation/UI).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformKind {
    /// Input Device Transform: source → ACES working space.
    Idt,
    /// Look Modification Transform: ACES → ACES (creative look).
    Lmt,
    /// Output Transform (RRT+ODT): ACES → display.
    OutputTransform,
    /// ASC Color Decision List: slope/offset/power/saturation.
    Cdl,
    /// Color Space Conversion: any → any.
    Csc,
}

/// A registered color transform.
#[derive(Debug, Clone)]
pub struct ColorTransform {
    pub name: String,
    pub kind: TransformKind,
    pub source_space: ColorSpace,
    pub target_space: ColorSpace,
    pub inner: ColorTransformInner,
}

/// The actual transform implementation.
#[derive(Debug, Clone)]
pub enum ColorTransformInner {
    /// Color space conversion via color_math.
    ColorSpaceConvert { from: ColorSpace, to: ColorSpace },
    /// LMT (3D LUT, analytical, or chain).
    Lmt(Lmt),
    /// ASC CDL: per-channel slope/offset/power + saturation.
    Cdl {
        slope: [f32; 3],
        offset: [f32; 3],
        power: [f32; 3],
        saturation: f32,
    },
    /// ACES 2.0 Output Transform — algorithmic, not expressible as CLF.
    /// Parametric by peak luminance and limiting primaries.
    Aces2OutputTransform(crate::aces2::Aces2OtParams),
}

impl ColorTransform {
    /// Apply the transform to f32 RGBA pixel data in-place.
    pub fn apply(&self, pixels: &mut [f32]) {
        match &self.inner {
            ColorTransformInner::ColorSpaceConvert { from, to } => {
                color_math::convert_color_space(pixels, *from, *to);
            }
            ColorTransformInner::Lmt(lmt) => {
                let out = lmt.apply(pixels);
                pixels.copy_from_slice(&out);
            }
            ColorTransformInner::Cdl { slope, offset, power, saturation } => {
                let sat = *saturation;
                for px in pixels.chunks_exact_mut(4) {
                    // SOP per channel
                    for c in 0..3 {
                        px[c] = ((px[c] * slope[c] + offset[c]).max(0.0)).powf(power[c]);
                    }
                    // Saturation via Rec.709 luma
                    if (sat - 1.0).abs() > 1e-6 {
                        let luma = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
                        for c in 0..3 {
                            px[c] = luma + sat * (px[c] - luma);
                        }
                    }
                }
            }
            ColorTransformInner::Aces2OutputTransform(params) => {
                crate::aces2::output_transform_fwd(pixels, params);
            }
        }
    }
}

// ─── Built-in Presets ──────────────────────────────────────────────────────

/// Preset info for discovery.
#[derive(Debug, Clone)]
pub struct TransformPresetInfo {
    pub name: &'static str,
    pub display_name: &'static str,
    pub kind: TransformKind,
    pub source_space: &'static str,
    pub target_space: &'static str,
    pub vendor: &'static str,
    pub description: &'static str,
}

/// All built-in transform presets.
pub fn preset_list() -> Vec<TransformPresetInfo> {
    let mut presets = vec![
        // IDTs — source → ACES (one-directional)
        TransformPresetInfo { name: "idt-srgb", display_name: "sRGB (IDT)", kind: TransformKind::Idt, source_space: "sRGB", target_space: "ACEScg", vendor: "Academy", description: "sRGB/Rec.709 input to ACEScg" },
        TransformPresetInfo { name: "idt-rec709", display_name: "Rec.709 (IDT)", kind: TransformKind::Idt, source_space: "Rec.709", target_space: "ACEScg", vendor: "Academy", description: "Rec.709 input to ACEScg" },
        TransformPresetInfo { name: "idt-rec2020", display_name: "Rec.2020 (IDT)", kind: TransformKind::Idt, source_space: "Rec.2020", target_space: "ACEScg", vendor: "Academy", description: "Rec.2020 input to ACEScg" },
        TransformPresetInfo { name: "idt-p3", display_name: "Display P3 (IDT)", kind: TransformKind::Idt, source_space: "Display P3", target_space: "ACEScg", vendor: "Academy", description: "Display P3 input to ACEScg" },
        // Output Transforms — ACES → display (one-directional)
        TransformPresetInfo { name: "ot-srgb", display_name: "sRGB 100 nits (OT)", kind: TransformKind::OutputTransform, source_space: "ACEScg", target_space: "sRGB", vendor: "Academy", description: "ACEScg to sRGB display" },
        TransformPresetInfo { name: "ot-rec709", display_name: "Rec.709 100 nits (OT)", kind: TransformKind::OutputTransform, source_space: "ACEScg", target_space: "Rec.709", vendor: "Academy", description: "ACEScg to Rec.709 display" },
        TransformPresetInfo { name: "ot-rec2020", display_name: "Rec.2020 100 nits (OT)", kind: TransformKind::OutputTransform, source_space: "ACEScg", target_space: "Rec.2020", vendor: "Academy", description: "ACEScg to Rec.2020 display" },
        TransformPresetInfo { name: "ot-p3", display_name: "Display P3 100 nits (OT)", kind: TransformKind::OutputTransform, source_space: "ACEScg", target_space: "Display P3", vendor: "Academy", description: "ACEScg to Display P3 display" },
        // CSCs — bidirectional working space conversions
        TransformPresetInfo { name: "csc-acescg-to-cct", display_name: "ACEScg to ACEScct", kind: TransformKind::Csc, source_space: "ACEScg", target_space: "ACEScct", vendor: "Academy", description: "Linear to log grading space" },
        TransformPresetInfo { name: "csc-acescct-to-cg", display_name: "ACEScct to ACEScg", kind: TransformKind::Csc, source_space: "ACEScct", target_space: "ACEScg", vendor: "Academy", description: "Log to linear compositing space" },
        // Utility
        TransformPresetInfo { name: "identity", display_name: "Identity", kind: TransformKind::Lmt, source_space: "any", target_space: "any", vendor: "Academy", description: "Passthrough (no-op)" },
    ];
    // Add camera IDT presets
    presets.extend(crate::camera_idt::camera_preset_list());
    presets
}

/// Load a built-in preset by name.
/// Helper to build a CSC preset.
fn csc_preset(name: &str, from: ColorSpace, to: ColorSpace, kind: TransformKind) -> ColorTransform {
    ColorTransform {
        name: name.into(), kind,
        source_space: from, target_space: to,
        inner: ColorTransformInner::ColorSpaceConvert { from, to },
    }
}

pub fn load_preset(name: &str) -> Result<ColorTransform, PipelineError> {
    match name {
        // IDTs — source → ACEScg
        "idt-srgb" => Ok(csc_preset(name, ColorSpace::Srgb, ColorSpace::AcesCg, TransformKind::Idt)),
        "idt-rec709" => Ok(csc_preset(name, ColorSpace::Rec709, ColorSpace::AcesCg, TransformKind::Idt)),
        "idt-rec2020" => Ok(csc_preset(name, ColorSpace::Rec2020, ColorSpace::AcesCg, TransformKind::Idt)),
        "idt-p3" => Ok(csc_preset(name, ColorSpace::DisplayP3, ColorSpace::AcesCg, TransformKind::Idt)),
        // Output Transforms — ACEScg → display (ACES 2.0 algorithmic)
        "ot-srgb" => Ok(ColorTransform {
            name: name.into(), kind: TransformKind::OutputTransform,
            source_space: ColorSpace::AcesCg, target_space: ColorSpace::Srgb,
            inner: ColorTransformInner::Aces2OutputTransform(crate::aces2::init_aces2_ot_params(
                100.0, crate::aces2::LimitingPrimaries::Rec709, crate::aces2::Eotf::Srgb,
            )),
        }),
        "ot-rec709" => Ok(ColorTransform {
            name: name.into(), kind: TransformKind::OutputTransform,
            source_space: ColorSpace::AcesCg, target_space: ColorSpace::Rec709,
            inner: ColorTransformInner::Aces2OutputTransform(crate::aces2::init_aces2_ot_params(
                100.0, crate::aces2::LimitingPrimaries::Rec709, crate::aces2::Eotf::Bt1886,
            )),
        }),
        "ot-rec2020" => Ok(ColorTransform {
            name: name.into(), kind: TransformKind::OutputTransform,
            source_space: ColorSpace::AcesCg, target_space: ColorSpace::Rec2020,
            inner: ColorTransformInner::Aces2OutputTransform(crate::aces2::init_aces2_ot_params(
                100.0, crate::aces2::LimitingPrimaries::Rec2020, crate::aces2::Eotf::Bt1886,
            )),
        }),
        "ot-p3" => Ok(ColorTransform {
            name: name.into(), kind: TransformKind::OutputTransform,
            source_space: ColorSpace::AcesCg, target_space: ColorSpace::DisplayP3,
            inner: ColorTransformInner::Aces2OutputTransform(crate::aces2::init_aces2_ot_params(
                100.0, crate::aces2::LimitingPrimaries::P3, crate::aces2::Eotf::Srgb,
            )),
        }),
        // CSCs — working space conversions
        "csc-acescg-to-cct" => Ok(csc_preset(name, ColorSpace::AcesCg, ColorSpace::AcesCct, TransformKind::Csc)),
        "csc-acescct-to-cg" => Ok(csc_preset(name, ColorSpace::AcesCct, ColorSpace::AcesCg, TransformKind::Csc)),
        // Utility
        "identity" => Ok(csc_preset(name, ColorSpace::Linear, ColorSpace::Linear, TransformKind::Lmt)),
        // Camera IDTs
        name if name.starts_with("idt-arri-") || name.starts_with("idt-sony-")
            || name.starts_with("idt-red-") || name.starts_with("idt-bmd-") => {
            crate::camera_idt::load_camera_preset(name)
        }
        _ => Err(PipelineError::InvalidParams(format!("unknown transform preset: {name}"))),
    }
}

/// Parse a color transform from file data.
pub fn parse_transform(data: &[u8], format_hint: Option<&str>) -> Result<ColorTransform, PipelineError> {
    let text = std::str::from_utf8(data)
        .map_err(|_| PipelineError::InvalidParams("transform data is not valid UTF-8".into()))?;

    // Auto-detect format
    let format = format_hint.unwrap_or_else(|| {
        if text.contains("LUT_3D_SIZE") || text.contains("LUT_1D_SIZE") {
            "cube"
        } else if text.contains("<ProcessList") {
            "clf"
        } else if text.contains("<ColorCorrection") || text.contains("<ColorDecisionList")
            || text.contains("<ColorCorrectionCollection") || text.contains("<SOPNode") {
            "cdl"
        } else if text.trim_start().starts_with("<?xml") {
            // Generic XML — could be CLF or CDL, check further
            if text.contains("<ProcessList") { "clf" } else { "cdl" }
        } else {
            "unknown"
        }
    });

    match format {
        "cube" => {
            let lmt = crate::lmt::parse_cube(text)?;
            Ok(ColorTransform {
                name: "custom-lut".into(),
                kind: TransformKind::Lmt,
                source_space: ColorSpace::Linear,
                target_space: ColorSpace::Linear,
                inner: ColorTransformInner::Lmt(lmt),
            })
        }
        "clf" => {
            let lmt = crate::lmt::parse_clf(text)?;
            Ok(ColorTransform {
                name: "custom-clf".into(),
                kind: TransformKind::Lmt,
                source_space: ColorSpace::Linear,
                target_space: ColorSpace::Linear,
                inner: ColorTransformInner::Lmt(lmt),
            })
        }
        "cdl" => {
            let cdl_list = crate::cdl::parse_cdl(text)?;
            let cdl = &cdl_list[0]; // Use first correction
            Ok(ColorTransform {
                name: cdl.id.clone().unwrap_or_else(|| "custom-cdl".into()),
                kind: TransformKind::Cdl,
                source_space: ColorSpace::Linear,
                target_space: ColorSpace::Linear,
                inner: ColorTransformInner::Cdl {
                    slope: cdl.slope,
                    offset: cdl.offset,
                    power: cdl.power,
                    saturation: cdl.saturation,
                },
            })
        }
        _ => Err(PipelineError::InvalidParams(format!("unsupported transform format: {format}"))),
    }
}

// ─── Graph Node ────────────────────────────────────────────────────────────

/// Color transform node in the pipeline graph.
pub struct ColorTransformNode {
    upstream: u32,
    info: NodeInfo,
    transform: ColorTransform,
}

impl ColorTransformNode {
    pub fn new(upstream: u32, upstream_info: NodeInfo, transform: ColorTransform) -> Self {
        let info = NodeInfo {
            width: upstream_info.width,
            height: upstream_info.height,
            color_space: transform.target_space,
        };
        Self { upstream, info, transform }
    }
}

impl Node for ColorTransformNode {
    fn info(&self) -> NodeInfo { self.info.clone() }

    fn compute(&self, request: Rect, upstream: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
        let mut pixels = upstream.request(self.upstream, request)?;
        self.transform.apply(&mut pixels);
        Ok(pixels)
    }

    fn upstream_ids(&self) -> Vec<u32> { vec![self.upstream] }

    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities::default() // CPU only for now; GPU shader in future track
    }

    fn input_rect(&self, output: Rect, _bounds_w: u32, _bounds_h: u32) -> InputRectEstimate {
        InputRectEstimate::Exact(output) // point operation, no spatial expansion
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_list_has_entries() {
        let presets = preset_list();
        assert!(presets.len() >= 10);
        assert!(presets.iter().any(|p| p.name == "idt-srgb"));
        assert!(presets.iter().any(|p| p.name == "ot-srgb"));
        assert!(presets.iter().any(|p| p.name == "identity"));
    }

    #[test]
    fn load_all_presets() {
        for preset in preset_list() {
            let t = load_preset(preset.name);
            assert!(t.is_ok(), "Failed to load preset: {}", preset.name);
        }
    }

    #[test]
    fn unknown_preset_returns_error() {
        assert!(load_preset("nonexistent").is_err());
    }

    #[test]
    fn identity_transform_is_noop() {
        let t = load_preset("identity").unwrap();
        let mut pixels = vec![0.5, 0.3, 0.8, 1.0];
        let orig = pixels.clone();
        t.apply(&mut pixels);
        for i in 0..4 {
            assert!((pixels[i] - orig[i]).abs() < 1e-6, "ch{i}");
        }
    }

    #[test]
    fn idt_then_ot_produces_valid_output() {
        let idt = load_preset("idt-srgb").unwrap();
        let odt = load_preset("ot-srgb").unwrap();
        let mut pixels = vec![0.5, 0.3, 0.8, 1.0];
        idt.apply(&mut pixels);
        // Pixels should have changed (sRGB → ACEScg)
        odt.apply(&mut pixels);
        // Should be approximately back to original
        // OT includes tone mapping, so output won't match input exactly.
        // Verify output is valid display values (0-1 after sRGB encoding).
        for i in 0..3 {
            assert!(pixels[i] >= 0.0 && pixels[i] <= 1.0,
                "ch{i} out of range: {}", pixels[i]);
        }
        assert!((pixels[3] - 1.0).abs() < 1e-6, "alpha preserved");
    }

    #[test]
    fn cdl_transform_applies() {
        let t = ColorTransform {
            name: "test-cdl".into(),
            kind: TransformKind::Cdl,
            source_space: ColorSpace::Linear,
            target_space: ColorSpace::Linear,
            inner: ColorTransformInner::Cdl {
                slope: [1.2, 1.0, 0.8],
                offset: [0.01, 0.0, -0.01],
                power: [1.0, 1.0, 1.0],
                saturation: 1.0,
            },
        };
        let mut pixels = vec![0.5, 0.5, 0.5, 1.0];
        t.apply(&mut pixels);
        assert!((pixels[0] - 0.61).abs() < 0.01); // 0.5 * 1.2 + 0.01
        assert!((pixels[1] - 0.50).abs() < 0.01); // 0.5 * 1.0 + 0.0
        assert!((pixels[2] - 0.39).abs() < 0.01); // 0.5 * 0.8 - 0.01
    }
}
