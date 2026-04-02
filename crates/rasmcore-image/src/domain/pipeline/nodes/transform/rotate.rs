use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform;
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

use super::affine::AffineOp;

/// Rotate node.
pub struct RotateNode {
    upstream: u32,
    rotation: Rotation,
    source_info: ImageInfo,
}

impl RotateNode {
    pub fn new(upstream: u32, source_info: ImageInfo, rotation: Rotation) -> Self {
        Self {
            upstream,
            rotation,
            source_info,
        }
    }
}

impl AffineOp for RotateNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        let w = self.source_info.width as f64;
        let h = self.source_info.height as f64;
        let (mat, ow, oh) = match self.rotation {
            Rotation::R90 => (
                [0.0, -1.0, h, 1.0, 0.0, 0.0],
                self.source_info.height,
                self.source_info.width,
            ),
            Rotation::R180 => (
                [-1.0, 0.0, w, 0.0, -1.0, h],
                self.source_info.width,
                self.source_info.height,
            ),
            Rotation::R270 => (
                [0.0, 1.0, 0.0, -1.0, 0.0, w],
                self.source_info.height,
                self.source_info.width,
            ),
        };
        (mat, ow, oh)
    }
}

#[rasmcore_macros::register_transform(name = "rotate")]
impl ImageNode for RotateNode {
    fn info(&self) -> ImageInfo {
        let (w, h) = match self.rotation {
            Rotation::R90 | Rotation::R270 => (self.source_info.height, self.source_info.width),
            Rotation::R180 => (self.source_info.width, self.source_info.height),
        };
        ImageInfo {
            width: w,
            height: h,
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
        let result = transform::rotate(&src_pixels, &self.source_info, self.rotation)?;
        Ok(result.pixels)
    }

    fn input_rect(&self, _output: Rect, _bounds_w: u32, _bounds_h: u32) -> Rect {
        Rect::new(0, 0, self.source_info.width, self.source_info.height)
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

    fn derive_metadata(
        &self,
        upstream: &rasmcore_pipeline::Metadata,
    ) -> Option<rasmcore_pipeline::Metadata> {
        let info = self.info();
        if info.width != self.source_info.width || info.height != self.source_info.height {
            let mut meta = upstream.clone();
            meta.set("width", rasmcore_pipeline::MetadataValue::Int(info.width as i64));
            meta.set("height", rasmcore_pipeline::MetadataValue::Int(info.height as i64));
            Some(meta)
        } else {
            None
        }
    }
}
