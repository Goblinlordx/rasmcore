//! Domain-level types for image processing.
//! These are independent of WIT — the adapter layer converts to/from WIT types.

/// Pixel format of image data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Bgr8,
    Bgra8,
    Gray8,
    Gray16,
    Rgb16,
    Rgba16,
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
    ProPhotoRgb,
    AdobeRgb,
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
    /// Embedded ICC color profile (opaque binary blob).
    /// Present when the source image contains an ICC profile (e.g., JPEG APP2, PNG iCCP).
    pub icc_profile: Option<Vec<u8>>,
}

// ─── Multi-Frame / Multi-Page Types ─────────────────────────────────────────

/// Frame disposal method for animated images (GIF, WebP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisposalMethod {
    /// No disposal — leave frame in place.
    None,
    /// Restore the area covered by the frame to the background.
    Background,
    /// Restore the area covered by the frame to the previous frame.
    Previous,
}

/// Per-frame metadata for multi-frame/multi-page images.
#[derive(Debug, Clone)]
pub struct FrameInfo {
    /// Zero-based frame index.
    pub index: u32,
    /// Frame delay in milliseconds (0 for non-animated formats like TIFF).
    pub delay_ms: u32,
    /// Disposal method (only meaningful for animated GIF/WebP).
    pub disposal: DisposalMethod,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Horizontal offset of the frame within the canvas.
    pub x_offset: u32,
    /// Vertical offset of the frame within the canvas.
    pub y_offset: u32,
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
