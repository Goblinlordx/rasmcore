//! Reusable parameter types for filter ConfigParams.
//!
//! These types carry a type-level hint via `#[config_hint("rc.*")]` so that
//! any ConfigParams struct embedding them automatically propagates the hint
//! to the UI manifest — no per-field annotation needed.
//!
//! # Example
//!
//! ```ignore
//! #[derive(ConfigParams)]
//! pub struct DrawLineParams {
//!     pub color: ColorRgba,  // hint "rc.color_rgba" auto-propagated
//!     #[param(min = 0.5, max = 100.0, step = 0.5, default = 2.0)]
//!     pub width: f32,
//! }
//! ```

/// RGBA color with 8-bit channels (0-255).
#[derive(rasmcore_macros::ConfigParams, Clone)]
#[config_hint("rc.color_rgba")]
pub struct ColorRgba {
    /// Red channel
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub r: u8,
    /// Green channel
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub g: u8,
    /// Blue channel
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub b: u8,
    /// Alpha channel
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub a: u8,
}

/// 2D point coordinate.
///
/// Used for polygon vertices and other operations that take point lists.
/// Codegen maps `&[Point2D]` → WIT `list<point2d>` → SDK `[x, y][]`.
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

/// RGB color with 8-bit channels (0-255).
#[derive(rasmcore_macros::ConfigParams, Clone)]
#[config_hint("rc.color_rgb")]
pub struct ColorRgb {
    /// Red channel
    #[param(min = 0, max = 255, step = 1, default = 128)]
    pub r: u8,
    /// Green channel
    #[param(min = 0, max = 255, step = 1, default = 128)]
    pub g: u8,
    /// Blue channel
    #[param(min = 0, max = 255, step = 1, default = 128)]
    pub b: u8,
}
