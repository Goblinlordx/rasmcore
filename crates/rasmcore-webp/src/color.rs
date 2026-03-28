//! RGB to YUV420 color space conversion.
//!
//! Converts input pixels to the YUV420 format used by VP8:
//! - Y (luma): full resolution, one value per pixel
//! - U, V (chroma): half resolution in both dimensions

/// YUV420 image buffer.
#[derive(Debug)]
pub struct YuvImage {
    pub width: u32,
    pub height: u32,
    /// Luma plane (width × height bytes).
    pub y: Vec<u8>,
    /// Chroma-U plane ((width/2) × (height/2) bytes).
    pub u: Vec<u8>,
    /// Chroma-V plane ((width/2) × (height/2) bytes).
    pub v: Vec<u8>,
}

/// Convert RGB8 pixels to YUV420.
///
/// Uses BT.601 coefficients (matching VP8/libwebp):
///   Y =  66*R + 129*G +  25*B + 128 >> 8 + 16
///   U = -38*R -  74*G + 112*B + 128 >> 8 + 128
///   V = 112*R -  94*G -  18*B + 128 >> 8 + 128
pub fn rgb_to_yuv420(_pixels: &[u8], _width: u32, _height: u32) -> YuvImage {
    // TODO: Implement in color conversion track
    todo!("RGB to YUV420 conversion")
}

/// Convert RGBA8 pixels to YUV420 (alpha channel discarded).
pub fn rgba_to_yuv420(_pixels: &[u8], _width: u32, _height: u32) -> YuvImage {
    // TODO: Implement in color conversion track
    todo!("RGBA to YUV420 conversion")
}
