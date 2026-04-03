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
pub enum ColorSpace {
    /// Linear sRGB (D65). The default working space for VFX operations.
    /// Blur, composite, resize — all should operate in linear light.
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

impl Default for ColorSpace {
    fn default() -> Self {
        ColorSpace::Linear
    }
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
}
