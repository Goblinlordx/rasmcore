//! ICC color conversion pipeline node.

use crate::domain::color;
use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::types::{ColorSpace, ImageInfo};
use rasmcore_pipeline::{Overlap, Rect};

/// Pipeline node that converts pixels from an ICC profile's color space to sRGB.
pub struct IccToSrgbNode {
    upstream: u32,
    source_info: ImageInfo,
    icc_profile: Vec<u8>,
}

impl IccToSrgbNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        icc_profile: Vec<u8>,
    ) -> Result<Self, ImageError> {
        // Validate profile is parseable at construction time
        moxcms::ColorProfile::new_from_slice(&icc_profile)
            .map_err(|e| ImageError::InvalidInput(format!("invalid ICC profile: {e:?}")))?;

        Ok(Self {
            upstream,
            source_info,
            icc_profile,
        })
    }
}

impl ImageNode for IccToSrgbNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            color_space: ColorSpace::Srgb,
            ..self.source_info.clone()
        }
    }

    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let full_src = Rect::new(0, 0, self.source_info.width, self.source_info.height);
        let src_pixels = upstream_fn(self.upstream, full_src)?;
        color::icc_to_srgb(&src_pixels, &self.source_info, &self.icc_profile)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}
