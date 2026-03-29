//! Shared integer DCT/DST transforms for image codecs.
//!
//! Provides forward and inverse transforms for block sizes 4x4 through 32x32,
//! used by JPEG (8x8), WebP (4x4), and HEVC (4-32x32).
//!
//! All transforms use separable row-column decomposition with integer arithmetic.
//! HEVC-spec butterfly structures are used for all sizes (ITU-T H.265 Section 8.6.4.2).

mod dct16;
mod dct32;
mod dct4;
mod dct8;
mod dst4;

pub use dct4::{forward_dct_4x4, inverse_dct_4x4};
pub use dct8::{forward_dct_8x8, inverse_dct_8x8};
pub use dct16::{forward_dct_16x16, inverse_dct_16x16};
pub use dct32::{forward_dct_32x32, inverse_dct_32x32};
pub use dst4::{forward_dst_4x4, inverse_dst_4x4};
