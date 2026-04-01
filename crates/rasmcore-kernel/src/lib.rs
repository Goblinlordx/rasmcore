//! rasmcore-kernel — Shared computation kernels for image processing.
//!
//! Pure math with no image-type dependencies. Provides:
//! - Color space conversions (sRGB, LAB, OKLab, ProPhoto, Adobe RGB, Bradford)
//! - LUT composition (1D point operations)
//! - 3D color lookup table infrastructure (CLUT)

#![allow(clippy::excessive_precision)]

pub mod color_lut;
pub mod color_spaces;
pub mod lut;
