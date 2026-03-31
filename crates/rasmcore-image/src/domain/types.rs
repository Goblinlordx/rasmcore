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
    Cmyk8,
    Cmyka8,
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
#[derive(Debug, Clone, Copy)]
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

/// Frame selection mode for multi-frame pipeline sources.
#[derive(Debug, Clone)]
pub enum FrameSelection {
    /// Select a single frame — pipeline runs once, output is a single image.
    Single(u32),
    /// Select specific frames by index — pipeline runs once per frame.
    Pick(Vec<u32>),
    /// Select a contiguous range (start inclusive, end exclusive).
    Range(u32, u32),
    /// Select all frames in the source.
    All,
}

/// An ordered collection of frames with per-frame metadata.
///
/// Used as the output of sequence-mode pipeline execution and as input
/// to multi-frame encoders (animated GIF, multi-page TIFF).
#[derive(Debug)]
pub struct FrameSequence {
    pub frames: Vec<(DecodedImage, FrameInfo)>,
    pub canvas_width: u32,
    pub canvas_height: u32,
}

impl FrameSequence {
    /// Create an empty sequence with the given canvas dimensions.
    pub fn new(canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            frames: Vec::new(),
            canvas_width,
            canvas_height,
        }
    }

    /// Add a frame with metadata.
    pub fn push(&mut self, image: DecodedImage, info: FrameInfo) {
        self.frames.push((image, info));
    }

    /// Number of frames in the sequence.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Build a sequence by decoding all frames from image data.
    pub fn from_decode(data: &[u8]) -> Result<Self, super::error::ImageError> {
        let frames = super::decoder::decode_all_frames(data)?;
        let (cw, ch) = frames
            .first()
            .map(|(img, _)| (img.info.width, img.info.height))
            .unwrap_or((0, 0));
        Ok(Self {
            frames,
            canvas_width: cw,
            canvas_height: ch,
        })
    }
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
