use super::super::error::ImageError;
use super::super::metadata::ExifOrientation;
use super::super::types::{DecodedImage, FlipDirection, ImageInfo, Rotation};

/// Auto-orient an image by applying the EXIF orientation transform.
///
/// Maps EXIF orientation values (1-8) to the correct sequence of
/// rotation and flip operations.
pub fn auto_orient(
    pixels: &[u8],
    info: &ImageInfo,
    orientation: ExifOrientation,
) -> Result<DecodedImage, ImageError> {
    match orientation {
        ExifOrientation::Normal => Ok(DecodedImage {
            pixels: pixels.to_vec(),
            info: info.clone(),
            icc_profile: None,
        }),
        ExifOrientation::FlipHorizontal => super::flip(pixels, info, FlipDirection::Horizontal),
        ExifOrientation::Rotate180 => super::rotate(pixels, info, Rotation::R180),
        ExifOrientation::FlipVertical => super::flip(pixels, info, FlipDirection::Vertical),
        ExifOrientation::Transpose => {
            let rotated = super::rotate(pixels, info, Rotation::R270)?;
            super::flip(&rotated.pixels, &rotated.info, FlipDirection::Horizontal)
        }
        ExifOrientation::Rotate90 => super::rotate(pixels, info, Rotation::R90),
        ExifOrientation::Transverse => {
            let rotated = super::rotate(pixels, info, Rotation::R90)?;
            super::flip(&rotated.pixels, &rotated.info, FlipDirection::Horizontal)
        }
        ExifOrientation::Rotate270 => super::rotate(pixels, info, Rotation::R270),
    }
}

/// Auto-orient using EXIF metadata from encoded data.
///
/// Reads EXIF orientation from the encoded data and applies the correct
/// transform to the pixel data. If no EXIF is found or orientation is
/// normal, returns the pixels unchanged.
pub fn auto_orient_from_exif(
    pixels: &[u8],
    info: &ImageInfo,
    encoded_data: &[u8],
) -> Result<DecodedImage, ImageError> {
    let orientation = match super::super::metadata::read_exif(encoded_data) {
        Ok(meta) => meta.orientation.unwrap_or(ExifOrientation::Normal),
        Err(_) => ExifOrientation::Normal,
    };
    auto_orient(pixels, info, orientation)
}
