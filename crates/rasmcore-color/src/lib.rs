//! rasmcore-color — Parameterized color space conversion.
//!
//! Shared by rasmcore codec crates (rasmcore-jpeg, rasmcore-webp, etc.).
//! Supports BT.601, BT.709, and BT.2020 coefficient matrices.
//! All arithmetic is integer-only — no floating point in hot paths.
//!
//! # Usage
//!
//! ```
//! use rasmcore_color::{rgb_to_ycbcr_420, ColorMatrix, YuvImage};
//!
//! let pixels = vec![128u8; 16 * 16 * 3]; // 16x16 RGB8
//! let yuv = rgb_to_ycbcr_420(&pixels, 16, 16, &ColorMatrix::BT601);
//! assert_eq!(yuv.y.len(), 256);
//! ```

mod convert;
mod matrix;
mod types;

pub use convert::{
    gray_to_y, rgb_to_ycbcr, rgb_to_ycbcr_420, rgba_to_ycbcr_420, ycbcr_to_rgb,
};
pub use matrix::ColorMatrix;
pub use types::{ChromaSubsampling, YuvImage};
