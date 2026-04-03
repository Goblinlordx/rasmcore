//! V2 codec implementations — unified f32 decode/encode for all image formats.
//!
//! Every decoder outputs f32 RGBA. Every encoder accepts f32 RGBA and handles
//! view transform (gamma encoding) and quantization internally.
//!
//! # Color Space Convention
//!
//! - sRGB formats (JPEG, PNG, WebP, GIF, BMP, QOI, ICO, TGA, TIFF, DDS, PNM):
//!   Decoders output f32 in sRGB color space. The pipeline's promote node
//!   linearizes to Linear for processing.
//!
//! - Linear formats (EXR, HDR, FITS):
//!   Decoders output f32 in Linear color space directly. No promote needed.
//!
//! - Encoders: sRGB encoders apply linear→sRGB gamma + u8 quantization.
//!   Linear encoders write f32 values directly.

pub mod convert;
pub mod decoders;
pub mod encoders;

// Re-export dispatch functions
pub use decoders::{decode, decode_with_hint, detect_format};
pub use encoders::encode;
