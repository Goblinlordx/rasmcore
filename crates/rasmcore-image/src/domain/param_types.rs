//! Reusable parameter types for domain operations.

/// RGBA color with 8-bit channels (0-255).
#[derive(Debug, Clone, Copy)]
pub struct ColorRgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// 2D point coordinate.
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

/// RGB color with 8-bit channels (0-255).
#[derive(Debug, Clone, Copy)]
pub struct ColorRgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}
