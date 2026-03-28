//! Shared types for color conversion.

/// YCbCr image with separate luma and chroma planes.
#[derive(Debug, Clone)]
pub struct YuvImage {
    pub width: u32,
    pub height: u32,
    /// Luma plane (width x height bytes for 4:4:4, same for 4:2:0).
    pub y: Vec<u8>,
    /// Chroma-blue plane. Size depends on subsampling.
    pub u: Vec<u8>,
    /// Chroma-red plane. Size depends on subsampling.
    pub v: Vec<u8>,
}

/// Chroma subsampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:4:4 — full resolution chroma (same size as luma).
    Full,
    /// 4:2:0 — half resolution in both dimensions (2x2 block averaging).
    Quarter,
}
