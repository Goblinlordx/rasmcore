use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform;
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

use super::affine::AffineOp;

/// Crop node.
pub struct CropNode {
    upstream: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    source_info: ImageInfo,
}

impl CropNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            upstream,
            x,
            y,
            width,
            height,
            source_info,
        }
    }
}

impl AffineOp for CropNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        // Crop = translation by (-x, -y), output is crop dimensions
        (
            [1.0, 0.0, -(self.x as f64), 0.0, 1.0, -(self.y as f64)],
            self.width,
            self.height,
        )
    }
}

#[rasmcore_macros::register_transform(name = "crop")]
impl ImageNode for CropNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.width,
            height: self.height,
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
        let result = transform::crop(
            &src_pixels,
            &self.source_info,
            self.x,
            self.y,
            self.width,
            self.height,
        )?;
        Ok(result.pixels)
    }

    fn input_rect(&self, output: Rect, _bounds_w: u32, _bounds_h: u32) -> Rect {
        // Offset output coords by crop origin to get source coords
        Rect::new(
            output.x + self.x,
            output.y + self.y,
            output.width,
            output.height,
        )
        .clamp(self.source_info.width, self.source_info.height)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
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
        let mut meta = upstream.clone();
        meta.set(
            "width",
            rasmcore_pipeline::MetadataValue::Int(self.width as i64),
        );
        meta.set(
            "height",
            rasmcore_pipeline::MetadataValue::Int(self.height as i64),
        );
        Some(meta)
    }
}
