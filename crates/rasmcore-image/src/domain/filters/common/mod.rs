//! Shared helper functions for filters.
//!
//! Pure utility functions with no ConfigParams dependencies.
//! Used by individual filter files via `use crate::domain::filters::common::*;`
//!
//! Organized into focused submodules; everything is re-exported for
//! backward compatibility.

pub mod analysis;
pub mod blending;
pub mod blur;
pub mod color;
pub mod contour;
pub mod convolution;
pub mod denoise;
pub mod distortion;
pub mod edge;
pub mod fusion;
pub mod morphology;
pub mod noise;
pub mod pixel_format;
pub mod pyramid;
pub mod threshold;
pub mod types;

// Re-export everything for backward compatibility.
// All filter files use `use crate::domain::filters::common::*;`
pub use analysis::*;
pub use blending::*;
pub use blur::*;
pub use color::*;
pub use contour::*;
pub use convolution::*;
pub use denoise::*;
pub use distortion::*;
pub use edge::*;
pub use fusion::*;
pub use morphology::*;
pub use noise::*;
pub use pixel_format::*;
pub use pyramid::*;
pub use threshold::*;
pub use types::*;

// Re-export upstream crate types used by all filters.
pub use crate::domain::error::ImageError;
pub use crate::domain::types::{ImageInfo, PixelFormat, ColorSpace, DecodedImage};
pub use rasmcore_pipeline::Rect;
pub use crate::domain::point_ops::LutPointOp;
pub use crate::domain::color_lut::{ColorLut3D, ColorLutOp, ColorOp, DEFAULT_CLUT_GRID};

// Re-export parent module items (ConfigParams, etc.)
#[allow(unused_imports)]
pub use super::*;
