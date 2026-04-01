use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform;
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

use super::affine::AffineOp;

/// Flip node.
pub struct FlipNode {
    upstream: u32,
    direction: FlipDirection,
    source_info: ImageInfo,
}

impl FlipNode {
    pub fn new(upstream: u32, source_info: ImageInfo, direction: FlipDirection) -> Self {
        Self {
            upstream,
            direction,
            source_info,
        }
    }
}

impl AffineOp for FlipNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        let w = self.source_info.width as f64;
        let h = self.source_info.height as f64;
        let mat = match self.direction {
            FlipDirection::Horizontal => [-1.0, 0.0, w, 0.0, 1.0, 0.0],
            FlipDirection::Vertical => [1.0, 0.0, 0.0, 0.0, -1.0, h],
        };
        (mat, self.source_info.width, self.source_info.height)
    }
}

#[rasmcore_macros::register_transform(name = "flip")]
impl ImageNode for FlipNode {
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
        let result = transform::flip(&src_pixels, &self.source_info, self.direction)?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }

    fn as_affine_op(&self) -> Option<([f64; 6], u32, u32)> {
        Some(self.to_affine())
    }
}
