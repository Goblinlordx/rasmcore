//! Mask generation, operations, and masked adjustment blending.
//!
//! Mask generators produce grayscale images for selective adjustments:
//! - gradient_linear / gradient_radial: parametric gradient masks
//! - luminance_range / color_range: image-derived masks
//! - from_path: brush stroke rasterization
//!
//! Mask operations: combine (add/subtract/intersect), invert, feather.
//! Masked blend: composites original + adjusted using mask weights.

mod color_range;
pub use color_range::*;
mod combine;
pub use combine::*;
mod feather;
pub use feather::*;
mod from_path;
pub use from_path::*;
mod gradient_linear;
pub use gradient_linear::*;
mod gradient_radial;
pub use gradient_radial::*;
mod invert;
pub use invert::*;
mod luminance_range;
pub use luminance_range::*;
mod masked_blend;
pub use masked_blend::*;

#[cfg(test)]
mod tests;
