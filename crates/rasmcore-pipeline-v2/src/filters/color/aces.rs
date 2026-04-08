// ─── ACES IDT/ODT ─────────────────────────────────────────────────────────
// Input Device Transform (source → ACEScct) and Output Display Transform
// (ACEScct → target). These are pipeline boundary nodes — application layer
// inserts them around grading operations for perceptually correct results.

use crate::node::PipelineError;
use crate::ops::Filter;

/// ACES Input Device Transform — converts source color space to ACEScct.
///
/// Insert after read() when ACES mode is enabled. All downstream point-op
/// filters (brightness, contrast, curves, etc.) then operate in ACEScct
/// log space, producing perceptually uniform results.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "aces_idt", category = "color", cost = "O(n)")]
pub struct AcesIdt {
    /// Source color space (0=sRGB, 1=Linear, 2=Rec709, 3=DisplayP3).
    #[param(min = 0.0, max = 3.0, step = 1.0, default = 0.0)]
    pub source: f32,
}

impl Filter for AcesIdt {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let from = match self.source as u32 {
            0 => crate::color_space::ColorSpace::Srgb,
            1 => crate::color_space::ColorSpace::Linear,
            _ => crate::color_space::ColorSpace::Srgb,
        };
        let mut out = input.to_vec();
        crate::color_math::convert_color_space(&mut out, from, crate::color_space::ColorSpace::AcesCct);
        Ok(out)
    }
}

/// ACES Output Display Transform — converts ACEScct to target color space.
///
/// Insert before write() or display. Converts from the ACEScct grading
/// space back to the target display/output color space.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "aces_odt", category = "color", cost = "O(n)")]
pub struct AcesOdt {
    /// Target color space (0=sRGB, 1=Linear, 2=Rec709, 3=DisplayP3).
    #[param(min = 0.0, max = 3.0, step = 1.0, default = 0.0)]
    pub target: f32,
}

impl Filter for AcesOdt {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let to = match self.target as u32 {
            0 => crate::color_space::ColorSpace::Srgb,
            1 => crate::color_space::ColorSpace::Linear,
            _ => crate::color_space::ColorSpace::Srgb,
        };
        let mut out = input.to_vec();
        crate::color_math::convert_color_space(&mut out, crate::color_space::ColorSpace::AcesCct, to);
        Ok(out)
    }
}

/// ACEScct → ACEScg convenience filter.
///
/// Converts from log grading space to linear AP1 for spatial operations
/// (blur, sharpen) that need linear light for physical correctness.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "aces_cct_to_cg", category = "color", cost = "O(n)")]
pub struct AcesCctToCg;

impl Filter for AcesCctToCg {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        crate::color_math::convert_color_space(
            &mut out,
            crate::color_space::ColorSpace::AcesCct,
            crate::color_space::ColorSpace::AcesCg,
        );
        Ok(out)
    }
}

/// ACEScg → ACEScct convenience filter.
///
/// Converts back from linear AP1 to log grading space after spatial operations.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "aces_cg_to_cct", category = "color", cost = "O(n)")]
pub struct AcesCgToCct;

impl Filter for AcesCgToCct {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        crate::color_math::convert_color_space(
            &mut out,
            crate::color_space::ColorSpace::AcesCg,
            crate::color_space::ColorSpace::AcesCct,
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aces_idt_odt_roundtrip() {
        // sRGB → ACEScct → sRGB should be near-identical
        let input = vec![0.2, 0.5, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let idt = AcesIdt { source: 0.0 }; // sRGB
        let odt = AcesOdt { target: 0.0 }; // sRGB

        let acescct = idt.compute(&input, 3, 1).unwrap();
        let roundtrip = odt.compute(&acescct, 3, 1).unwrap();

        for (i, (&orig, &rt)) in input.iter().zip(roundtrip.iter()).enumerate() {
            let diff = (orig - rt).abs();
            assert!(diff < 0.001, "pixel[{i}] roundtrip error: {orig} → {rt} (diff {diff})");
        }
    }

    #[test]
    fn aces_cct_cg_roundtrip() {
        // ACEScct → ACEScg → ACEScct should be near-identical
        let idt = AcesIdt { source: 0.0 };
        let to_cg = AcesCctToCg;
        let to_cct = AcesCgToCct;

        let input = vec![0.3, 0.6, 0.9, 1.0];
        let acescct = idt.compute(&input, 1, 1).unwrap();
        let acescg = to_cg.compute(&acescct, 1, 1).unwrap();
        let back = to_cct.compute(&acescg, 1, 1).unwrap();

        for (i, (&orig, &rt)) in acescct.iter().zip(back.iter()).enumerate() {
            let diff = (orig - rt).abs();
            assert!(diff < 0.0001, "cct↔cg roundtrip[{i}] error: {orig} → {rt} (diff {diff})");
        }
    }
}
