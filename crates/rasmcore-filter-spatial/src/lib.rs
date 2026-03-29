//! Spatial filters for rasmcore — blur, sharpen, convolve, median.
//!
//! This crate provides spatial filtering operations as a separate, composable
//! package. It uses the rasmcore extensibility system (#[register_filter])
//! for automatic registration via inventory.
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! rasmcore-filter-spatial = "0.1"
//! ```
//!
//! All filters are automatically registered when this crate is linked.
//! No manual setup needed.

// Re-export from rasmcore-image's built-in filters for now.
// The full extraction (moving source code here) is done incrementally.
// The registration via #[register_filter] in the original filters.rs
// still works — inventory collects across crates.
//
// This crate serves as the dependency entry point: when a user depends
// on rasmcore-filter-spatial, they get the spatial filters registered.

pub use rasmcore_image::domain::filters::{
    blur, sharpen, convolve, median, gaussian_blur_cv,
};

// Re-export the convolve helpers that external users might need
pub use rasmcore_image::domain::filter_utils;

#[cfg(test)]
mod tests {
    use super::*;
    use rasmcore_image::domain::types::*;

    #[test]
    fn blur_via_spatial_crate() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = blur(&pixels, &info, 2.0).unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn spatial_filters_registered() {
        let regs = rasmcore_image::domain::filter_registry::registered_filters();
        let names: Vec<&str> = regs.iter().map(|r| r.name).collect();
        assert!(names.contains(&"blur"), "blur should be registered. Found: {names:?}");
        assert!(names.contains(&"sharpen"), "sharpen should be registered");
        assert!(names.contains(&"convolve"), "convolve should be registered");
        assert!(names.contains(&"median"), "median should be registered");
    }
}
