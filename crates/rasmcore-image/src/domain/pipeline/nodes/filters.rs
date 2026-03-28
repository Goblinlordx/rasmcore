//! Filter nodes — wrap existing domain filter operations.

use crate::domain::error::ImageError;
use crate::domain::filters;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::types::*;
use rasmcore_pipeline::{Overlap, Rect};

macro_rules! simple_filter_node {
    ($name:ident, $param_type:ty, $fn_name:ident, $overlap_val:expr, $doc:expr) => {
        #[doc = $doc]
        pub struct $name {
            upstream: u32,
            param: $param_type,
            source_info: ImageInfo,
        }

        impl $name {
            pub fn new(upstream: u32, source_info: ImageInfo, param: $param_type) -> Self {
                Self {
                    upstream,
                    param,
                    source_info,
                }
            }
        }

        impl ImageNode for $name {
            fn info(&self) -> ImageInfo {
                self.source_info.clone()
            }

            fn compute_region(
                &self,
                _request: Rect,
                upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
            ) -> Result<Vec<u8>, ImageError> {
                let full_src = Rect::new(0, 0, self.source_info.width, self.source_info.height);
                let src_pixels = upstream_fn(self.upstream, full_src)?;
                filters::$fn_name(&src_pixels, &self.source_info, self.param)
            }

            fn overlap(&self) -> Overlap {
                $overlap_val
            }
            fn access_pattern(&self) -> AccessPattern {
                AccessPattern::LocalNeighborhood
            }
        }
    };
}

simple_filter_node!(
    BlurNode,
    f32,
    blur,
    Overlap::uniform(10),
    "Gaussian blur node."
);
simple_filter_node!(
    SharpenNode,
    f32,
    sharpen,
    Overlap::uniform(2),
    "Sharpen node."
);
simple_filter_node!(
    BrightnessNode,
    f32,
    brightness,
    Overlap::zero(),
    "Brightness adjustment node."
);
simple_filter_node!(
    ContrastNode,
    f32,
    contrast,
    Overlap::zero(),
    "Contrast adjustment node."
);

/// Grayscale node — changes pixel format to Gray8.
pub struct GrayscaleNode {
    upstream: u32,
    source_info: ImageInfo,
}

impl GrayscaleNode {
    pub fn new(upstream: u32, source_info: ImageInfo) -> Self {
        Self {
            upstream,
            source_info,
        }
    }
}

impl ImageNode for GrayscaleNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            format: PixelFormat::Gray8,
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
        let result = filters::grayscale(&src_pixels, &self.source_info)?;
        Ok(result.pixels)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}
