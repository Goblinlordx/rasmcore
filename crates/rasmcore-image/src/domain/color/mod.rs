//! ICC color profile operations and CMYK/RGB conversions.
//!
//! Uses moxcms (pure Rust, WASM-compatible) for ICC profile parsing and
//! color space transforms. This module provides:
//! - ICC profile extraction from JPEG/PNG raw bytes
//! - ICC-to-sRGB pixel conversion via moxcms transform
//! - CMYK <-> RGB conversion

mod conversions;
mod icc;

pub use conversions::*;
pub use icc::*;
