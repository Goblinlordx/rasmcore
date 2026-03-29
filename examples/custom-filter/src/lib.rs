//! Example custom filter crate — demonstrates third-party extensibility.
//!
//! This crate adds a custom "invert_red" filter to rasmcore without
//! modifying any rasmcore source code. Just depend on rasmcore-macros +
//! rasmcore-image, annotate your function, and it's registered.
//!
//! The registration is collected via `inventory` at link time across
//! crate boundaries.

use rasmcore_image::domain::error::ImageError;
use rasmcore_image::domain::types::{ImageInfo, PixelFormat};

/// Invert only the red channel — a trivial custom filter for testing.
///
/// Demonstrates that a third-party crate can register a filter using
/// the same `#[register_filter]` attribute as built-in filters.
#[rasmcore_macros::register_filter(name = "invert_red", category = "custom")]
pub fn invert_red(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    match info.format {
        PixelFormat::Rgb8 => {
            let mut result = pixels.to_vec();
            for chunk in result.chunks_exact_mut(3) {
                chunk[0] = 255 - chunk[0]; // invert R only
            }
            Ok(result)
        }
        PixelFormat::Rgba8 => {
            let mut result = pixels.to_vec();
            for chunk in result.chunks_exact_mut(4) {
                chunk[0] = 255 - chunk[0]; // invert R only
            }
            Ok(result)
        }
        _ => Err(ImageError::UnsupportedFormat(
            "invert_red requires RGB8 or RGBA8".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rasmcore_image::domain::types::ColorSpace;

    #[test]
    fn invert_red_works() {
        let pixels = vec![100u8, 150, 200, 50, 100, 150];
        let info = ImageInfo {
            width: 2,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = invert_red(&pixels, &info).unwrap();
        assert_eq!(result[0], 155); // 255 - 100
        assert_eq!(result[1], 150); // G unchanged
        assert_eq!(result[2], 200); // B unchanged
    }

    #[test]
    fn registration_is_discoverable() {
        let regs = rasmcore_image::domain::filter_registry::registered_filters();
        let found = regs.iter().any(|r| r.name == "invert_red");
        assert!(
            found,
            "invert_red should be registered via inventory. Found: {:?}",
            regs.iter().map(|r| r.name).collect::<Vec<_>>()
        );
    }
}
