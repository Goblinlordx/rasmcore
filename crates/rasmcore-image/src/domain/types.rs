/// Domain-level types for image processing.
/// These are independent of WIT — the adapter layer converts to/from WIT types.

/// Pixel format of image data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Bgr8,
    Bgra8,
    Gray8,
    Gray16,
    Yuv420p,
    Yuv422p,
    Yuv444p,
    Nv12,
}

/// Color space of image data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb,
    LinearSrgb,
    DisplayP3,
    Bt709,
    Bt2020,
}

/// Image metadata
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub color_space: ColorSpace,
}

/// A decoded image with pixel data and metadata
#[derive(Debug)]
pub struct DecodedImage {
    pub pixels: Vec<u8>,
    pub info: ImageInfo,
}

/// Resize algorithm
#[derive(Debug, Clone, Copy)]
pub enum ResizeFilter {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
}

/// Rotation amount
#[derive(Debug, Clone, Copy)]
pub enum Rotation {
    R90,
    R180,
    R270,
}

/// Flip direction
#[derive(Debug, Clone, Copy)]
pub enum FlipDirection {
    Horizontal,
    Vertical,
}
