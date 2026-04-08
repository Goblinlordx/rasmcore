//! Built-in color transform definitions for rasmcore.
//!
//! This crate contains the mathematical definitions (primaries, matrices,
//! transfer functions) for professional color spaces. Each transform source
//! has its own license — see the `licenses/` directory.
//!
//! The crate is pure data — no pipeline dependencies. The rasmcore-pipeline-v2
//! crate consumes these definitions to create ColorTransform resources.
//!
//! # Sources
//!
//! - **ACES:** A.M.P.A.S. (Academy), permissive license
//! - **ARRI:** Published specification (Section 4: "for 3rd Party implementations")
//! - **Blackmagic:** Published specification, approved in OCIO (BSD-3)

pub mod primaries;
pub mod transfer;
pub mod presets;

/// CIE xy chromaticity coordinates for RGB primaries + white point.
#[derive(Debug, Clone, Copy)]
pub struct Primaries {
    pub red: (f64, f64),
    pub green: (f64, f64),
    pub blue: (f64, f64),
    pub white: (f64, f64),
}

/// A 3x3 matrix (row-major) for RGB color space conversion.
pub type Mat3 = [f64; 9];

/// Transfer function definition.
#[derive(Debug, Clone)]
pub enum TransferFn {
    /// Linear (no encoding). Identity.
    Linear,
    /// sRGB EOTF (IEC 61966-2-1).
    Srgb,
    /// BT.1886 (pure 2.4 gamma).
    Bt1886,
    /// ACES CCT log (S-2016-001).
    AcesCct,
    /// ACES CC log (S-2008-001).
    AcesCc,
    /// ARRI LogC4 (single curve, no EI dependency).
    ArriLogC4,
    /// DaVinci Intermediate.
    DaVinciIntermediate,
    /// Generic piecewise log with constants.
    PiecewiseLog {
        /// Linear segment: out = lin_slope * in
        lin_slope: f64,
        /// Cut point (linear domain)
        lin_cut: f64,
        /// Log segment: out = log_coeff * (log_base.log(in + log_offset) + log_shift)
        log_base: f64,
        log_coeff: f64,
        log_offset: f64,
        log_shift: f64,
    },
}

/// A complete color space definition — primaries + transfer function.
#[derive(Debug, Clone)]
pub struct ColorSpaceDef {
    /// Identifier (e.g., "arri-logc4-awg4").
    pub id: &'static str,
    /// Human-readable name.
    pub display_name: &'static str,
    /// Vendor/source.
    pub vendor: &'static str,
    /// CIE xy primaries.
    pub primaries: Primaries,
    /// Transfer function (encoding).
    pub transfer: TransferFn,
    /// RGB-to-XYZ matrix (derived from primaries + white point).
    pub to_xyz: Mat3,
    /// XYZ-to-RGB matrix (inverse of to_xyz).
    pub from_xyz: Mat3,
}

/// A transform preset — what the pipeline exposes as a built-in.
#[derive(Debug, Clone)]
pub struct TransformPreset {
    /// Preset name used in get-transform (e.g., "idt-srgb").
    pub name: &'static str,
    /// Human-readable display name.
    pub display_name: &'static str,
    /// Transform kind: "idt", "ot", "csc", "lmt".
    pub kind: &'static str,
    /// Source color space id.
    pub source: &'static str,
    /// Target color space id.
    pub target: &'static str,
    /// Vendor/source.
    pub vendor: &'static str,
    /// Description.
    pub description: &'static str,
}
