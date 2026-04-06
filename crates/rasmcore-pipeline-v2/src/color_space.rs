//! Color space tracking for the V2 pipeline.
//!
//! Every node declares what color space its output is in. The graph walker
//! auto-inserts conversion nodes when a downstream node expects a different
//! color space from its upstream.

/// Color space of pixel data at a given point in the graph.
///
/// Tracked per-node in `NodeInfo`. The graph walker uses this to detect
/// mismatches and insert automatic conversions at node boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[derive(Default)]
pub enum ColorSpace {
    /// Linear sRGB (D65). The default working space for VFX operations.
    /// Blur, composite, resize — all should operate in linear light.
    #[default]
    Linear,

    /// sRGB with gamma encoding (IEC 61966-2-1). Raw decoded data from
    /// JPEG, PNG, etc. before linearization.
    Srgb,

    /// ACES2065-1 (AP0 primaries, D60). Archival interchange format.
    Aces2065_1,

    /// ACEScg (AP1 primaries, D60, linear). CG rendering working space.
    AcesCg,

    /// ACEScct (AP1 primaries, D60, log with toe). Color grading working space.
    /// Sliders feel "musical" — perceptually uniform for human adjustments.
    AcesCct,

    /// ACEScc (AP1 primaries, D60, pure log). Older grading space, no toe.
    AcesCc,

    /// Display P3 (D65, linear). Wide-gamut display space.
    DisplayP3,

    /// Rec. 709 (D65). Standard broadcast color space.
    Rec709,

    /// Rec. 2020 (D65). Wide-gamut HDR broadcast space.
    Rec2020,

    /// Unknown or unmanaged. Data passes through without conversion.
    /// Used for raw/untagged data or when color management is disabled.
    Unknown,
}


impl ColorSpace {
    /// True if this color space uses linear light (no transfer function).
    pub fn is_linear(self) -> bool {
        matches!(
            self,
            ColorSpace::Linear | ColorSpace::AcesCg | ColorSpace::Aces2065_1
        )
    }

    /// True if this color space uses logarithmic encoding (for grading).
    pub fn is_log(self) -> bool {
        matches!(self, ColorSpace::AcesCct | ColorSpace::AcesCc)
    }

    /// True if this space is perceptually uniform (log-encoded).
    ///
    /// In perceptually uniform spaces, equal numeric steps produce roughly
    /// equal perceived brightness changes — sliders feel proportional.
    /// ACEScct and ACEScc are the primary examples.
    pub fn is_perceptually_uniform(self) -> bool {
        self.is_log()
    }

    /// True if this space is scene-referred (values represent physical light ratios).
    ///
    /// Scene-referred spaces allow values >1.0 representing real-world
    /// brightness. ACES working spaces are scene-referred.
    /// Display-referred spaces (sRGB, Rec.709) are clamped 0–1.
    pub fn is_scene_referred(self) -> bool {
        matches!(
            self,
            ColorSpace::Linear
                | ColorSpace::AcesCg
                | ColorSpace::AcesCct
                | ColorSpace::AcesCc
                | ColorSpace::Aces2065_1
        )
    }

    /// Luminance coefficients derived from the color space's RGB primaries.
    ///
    /// These are the Y row of the RGB-to-XYZ matrix for this color space.
    /// Used for perceptual luminance: `Y = coeffs[0]*R + coeffs[1]*G + coeffs[2]*B`.
    ///
    /// Spaces sharing the same RGB primaries return identical coefficients
    /// regardless of transfer function (e.g., Linear and Srgb both use
    /// Rec.709 primaries).
    ///
    /// Source: CIE colorimetry — luminance coefficients are defined by the
    /// chromaticity coordinates of a color space's RGB primaries.
    pub fn luma_coefficients(self) -> [f32; 3] {
        match self {
            // Rec.709 primaries (sRGB, Linear sRGB, Rec.709)
            // Y row of SRGB_TO_XYZ_D65 matrix (color_math.rs:121)
            ColorSpace::Linear | ColorSpace::Srgb | ColorSpace::Rec709 => {
                [0.2126390, 0.7151687, 0.0721923]
            }

            // AP1 primaries (ACEScg, ACEScct, ACEScc)
            // Y row of AP1_TO_XYZ_D60 matrix (color_math.rs:163)
            ColorSpace::AcesCg | ColorSpace::AcesCct | ColorSpace::AcesCc => {
                [0.2722287, 0.6740818, 0.0536895]
            }

            // AP0 primaries (ACES2065-1)
            // Y row of AP0_TO_XYZ_D60 matrix (color_math.rs:149)
            // Note: blue coefficient is negative because AP0 primaries extend
            // beyond visible light. Filters should generally work in ACEScg,
            // not AP0, for luminance-dependent operations.
            ColorSpace::Aces2065_1 => [0.3439664, 0.7281661, -0.0721325],

            // Rec.2020 primaries (ITU-R BT.2020)
            ColorSpace::Rec2020 => [0.2627002, 0.6779981, 0.0593017],

            // Display P3 primaries (DCI-P3 D65)
            // Derived from P3 RGB-to-XYZ (D65) matrix Y row.
            ColorSpace::DisplayP3 => [0.2289746, 0.6917385, 0.0792869],

            // Unknown — default to Rec.709 (most common working assumption)
            ColorSpace::Unknown => [0.2126390, 0.7151687, 0.0721923],
        }
    }

    /// CIE xy chromaticity of the white point for this color space.
    ///
    /// Most spaces use D65 (daylight, ~6504K). ACES spaces use D60 (~6000K).
    pub fn white_point(self) -> (f32, f32) {
        match self {
            // D65 (CIE standard illuminant)
            ColorSpace::Linear
            | ColorSpace::Srgb
            | ColorSpace::Rec709
            | ColorSpace::Rec2020
            | ColorSpace::DisplayP3
            | ColorSpace::Unknown => (0.3127, 0.3290),

            // D60 (ACES illuminant, SMPTE ST 2065-1)
            ColorSpace::AcesCg
            | ColorSpace::AcesCct
            | ColorSpace::AcesCc
            | ColorSpace::Aces2065_1 => (0.32168, 0.33767),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_coefficients_match_xyz_matrix_y_row() {
        // Verify against the Y row of SRGB_TO_XYZ_D65 in color_math.rs
        let [r, g, b] = ColorSpace::Linear.luma_coefficients();
        assert!((r - 0.2126390).abs() < 1e-6);
        assert!((g - 0.7151687).abs() < 1e-6);
        assert!((b - 0.0721923).abs() < 1e-6);

        // Verify against AP1_TO_XYZ_D60 Y row
        let [r, g, b] = ColorSpace::AcesCg.luma_coefficients();
        assert!((r - 0.2722287).abs() < 1e-6);
        assert!((g - 0.6740818).abs() < 1e-6);
        assert!((b - 0.0536895).abs() < 1e-6);

        // Verify against AP0_TO_XYZ_D60 Y row (has negative blue)
        let [_, _, b] = ColorSpace::Aces2065_1.luma_coefficients();
        assert!(b < 0.0, "AP0 blue coefficient should be negative");
    }

    #[test]
    fn shared_primaries_return_identical_coefficients() {
        // Rec.709 primaries: Linear, Srgb, Rec709
        assert_eq!(
            ColorSpace::Linear.luma_coefficients(),
            ColorSpace::Srgb.luma_coefficients()
        );
        assert_eq!(
            ColorSpace::Linear.luma_coefficients(),
            ColorSpace::Rec709.luma_coefficients()
        );

        // AP1 primaries: AcesCg, AcesCct, AcesCc
        assert_eq!(
            ColorSpace::AcesCg.luma_coefficients(),
            ColorSpace::AcesCct.luma_coefficients()
        );
        assert_eq!(
            ColorSpace::AcesCg.luma_coefficients(),
            ColorSpace::AcesCc.luma_coefficients()
        );
    }

    #[test]
    fn luma_coefficients_sum_approximately_one() {
        for cs in [
            ColorSpace::Linear,
            ColorSpace::Srgb,
            ColorSpace::Rec709,
            ColorSpace::Rec2020,
            ColorSpace::AcesCg,
            ColorSpace::DisplayP3,
        ] {
            let [r, g, b] = cs.luma_coefficients();
            let sum = r + g + b;
            assert!(
                (sum - 1.0).abs() < 0.001,
                "{cs:?}: luma coefficients sum to {sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn different_primaries_return_different_coefficients() {
        assert_ne!(
            ColorSpace::Linear.luma_coefficients(),
            ColorSpace::AcesCg.luma_coefficients()
        );
        assert_ne!(
            ColorSpace::Linear.luma_coefficients(),
            ColorSpace::Rec2020.luma_coefficients()
        );
        assert_ne!(
            ColorSpace::AcesCg.luma_coefficients(),
            ColorSpace::DisplayP3.luma_coefficients()
        );
    }

    #[test]
    fn white_point_d65_vs_d60() {
        let (x65, y65) = ColorSpace::Linear.white_point();
        assert!((x65 - 0.3127).abs() < 1e-4);
        assert!((y65 - 0.3290).abs() < 1e-4);

        let (x60, y60) = ColorSpace::AcesCg.white_point();
        assert!((x60 - 0.32168).abs() < 1e-4);
        assert!((y60 - 0.33767).abs() < 1e-4);

        // D65 and D60 are different
        assert!((x65 - x60).abs() > 0.005);
    }

    #[test]
    fn scene_referred_classification() {
        assert!(ColorSpace::Linear.is_scene_referred());
        assert!(ColorSpace::AcesCg.is_scene_referred());
        assert!(ColorSpace::AcesCct.is_scene_referred());
        assert!(!ColorSpace::Srgb.is_scene_referred());
        assert!(!ColorSpace::Rec709.is_scene_referred());
        assert!(!ColorSpace::DisplayP3.is_scene_referred());
    }

    #[test]
    fn perceptually_uniform_classification() {
        assert!(ColorSpace::AcesCct.is_perceptually_uniform());
        assert!(ColorSpace::AcesCc.is_perceptually_uniform());
        assert!(!ColorSpace::Linear.is_perceptually_uniform());
        assert!(!ColorSpace::AcesCg.is_perceptually_uniform());
        assert!(!ColorSpace::Srgb.is_perceptually_uniform());
    }
}
