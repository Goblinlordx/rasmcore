//! Macroblock and sub-block layout for VP8 encoding.
//!
//! VP8 partitions images into 16×16 macroblocks. Each macroblock contains:
//! - 16 luma (Y) 4×4 sub-blocks
//! - 4 chroma-U 4×4 sub-blocks
//! - 4 chroma-V 4×4 sub-blocks
//! - 1 virtual Y2 block (4×4 DC coefficients from Y sub-blocks)

/// Information about a single macroblock.
#[derive(Debug, Clone)]
pub struct MacroblockInfo {
    /// Column index in macroblock grid.
    pub mb_x: u32,
    /// Row index in macroblock grid.
    pub mb_y: u32,
    /// Prediction mode for 16×16 luma.
    pub y_mode: u8,
    /// Prediction modes for 4×4 luma sub-blocks (if using B_PRED).
    pub b_modes: [u8; 16],
    /// Prediction mode for chroma.
    pub uv_mode: u8,
    /// Segment ID (0-3).
    pub segment: u8,
}

/// Calculate macroblock grid dimensions for a given image size.
pub fn mb_dimensions(width: u32, height: u32) -> (u32, u32) {
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    (mb_w, mb_h)
}
