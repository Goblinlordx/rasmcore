//! Histogram-based filters for rasmcore — equalize, normalize, auto-level, CLAHE.
pub use rasmcore_image::domain::filter_utils;
pub use rasmcore_image::domain::histogram::{auto_level, contrast_stretch, equalize, normalize};
