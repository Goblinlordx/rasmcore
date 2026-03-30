//! All standard rasmcore filters — convenience meta-crate.
//!
//! Depend on this to get all built-in filter categories:
//! - Spatial (blur, sharpen, convolve, median)
//! - Color (hue rotate, saturate, sepia, colorize)
//! - Edge (sobel, canny)
//! - Alpha (premultiply, flatten, blend modes)
//! - Histogram (equalize, normalize, CLAHE)
//! - Enhancement (dehaze, clarity, retinex)
//! - Advanced (bilateral, guided filter, NLM)
//!
//! Or depend on individual category crates for minimal builds.

pub use rasmcore_filter_advanced;
pub use rasmcore_filter_alpha;
pub use rasmcore_filter_color;
pub use rasmcore_filter_edge;
pub use rasmcore_filter_enhance;
pub use rasmcore_filter_histogram;
pub use rasmcore_filter_spatial;
