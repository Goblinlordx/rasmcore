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

// ─── Point-op nodes (LUT-fusible) ───────────────────────────────────────────

/// A pipeline node for any composable LUT-based point operation.
///
/// At execution, builds the LUT from the stored `PointOp` and applies it in one pass.
/// The pipeline optimizer can detect consecutive `PointOpNode`s and replace them
/// with a single `FusedLutNode` that applies one pre-composed LUT.
pub struct PointOpNode {
    upstream: u32,
    op: crate::domain::point_ops::PointOp,
    source_info: ImageInfo,
}

impl PointOpNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        op: crate::domain::point_ops::PointOp,
    ) -> Self {
        Self {
            upstream,
            op,
            source_info,
        }
    }
}

impl ImageNode for PointOpNode {
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
        let lut = crate::domain::point_ops::build_lut(&self.op);
        crate::domain::point_ops::apply_lut(&src_pixels, &self.source_info, &lut)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

/// A pipeline node holding a pre-composed LUT from fused point operations.
///
/// Created by the pipeline optimizer when it detects consecutive `PointOpNode`s.
/// Applies one LUT pass regardless of how many ops were fused.
pub struct FusedLutNode {
    upstream: u32,
    lut: [u8; 256],
    source_info: ImageInfo,
}

impl FusedLutNode {
    pub fn new(upstream: u32, source_info: ImageInfo, lut: [u8; 256]) -> Self {
        Self {
            upstream,
            lut,
            source_info,
        }
    }

    /// Create a fused node from multiple point operations.
    pub fn from_ops(
        upstream: u32,
        source_info: ImageInfo,
        ops: &[crate::domain::point_ops::PointOp],
    ) -> Self {
        use crate::domain::point_ops::{build_lut, compose_luts};
        let mut lut: [u8; 256] = std::array::from_fn(|i| i as u8); // identity
        for op in ops {
            let op_lut = build_lut(op);
            lut = compose_luts(&lut, &op_lut);
        }
        Self::new(upstream, source_info, lut)
    }
}

impl ImageNode for FusedLutNode {
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
        crate::domain::point_ops::apply_lut(&src_pixels, &self.source_info, &self.lut)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

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
