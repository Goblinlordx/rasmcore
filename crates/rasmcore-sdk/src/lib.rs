//! rasmcore Rust Native SDK — fluent builder API for image processing.
//!
//! This crate provides a zero-overhead Rust API wrapping rasmcore-image domain
//! functions. No WASM serialization — direct Rust linking.
//!
//! # Example
//!
//! ```ignore
//! use rasmcore_sdk::RcImage;
//!
//! let jpeg = RcImage::load(&png_bytes)?
//!     .blur(3.0)
//!     .resize(800, 600)?
//!     .brightness(0.1)?
//!     .to_jpeg(85)?;
//! ```

// Re-export key types for convenience
pub use rasmcore_image::domain::error::ImageError;
pub use rasmcore_image::domain::types::{ColorSpace, ImageInfo, PixelFormat};

// Include the auto-generated fluent API
include!(concat!(env!("OUT_DIR"), "/generated_sdk_rust.rs"));
