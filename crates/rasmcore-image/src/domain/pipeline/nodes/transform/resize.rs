use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform;
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

use super::affine::AffineOp;

/// Resize node.
pub struct ResizeNode {
    upstream: u32,
    target_width: u32,
    target_height: u32,
    filter: ResizeFilter,
    source_info: ImageInfo,
}

impl ResizeNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        width: u32,
        height: u32,
        filter: ResizeFilter,
    ) -> Self {
        Self {
            upstream,
            target_width: width,
            target_height: height,
            filter,
            source_info,
        }
    }
}

impl AffineOp for ResizeNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        let sx = self.target_width as f64 / self.source_info.width as f64;
        let sy = self.target_height as f64 / self.source_info.height as f64;
        (
            [sx, 0.0, 0.0, 0.0, sy, 0.0],
            self.target_width,
            self.target_height,
        )
    }
}

#[rasmcore_macros::register_transform(name = "resize")]
impl ImageNode for ResizeNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.target_width,
            height: self.target_height,
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
        let result = transform::resize(
            &src_pixels,
            &self.source_info,
            self.target_width,
            self.target_height,
            self.filter,
        )?;
        Ok(result.pixels)
    }

    fn input_rect(&self, output: Rect, _bounds_w: u32, _bounds_h: u32) -> Rect {
        let sx = self.source_info.width as f64 / self.target_width as f64;
        let sy = self.source_info.height as f64 / self.target_height as f64;
        // Map output rect to source coords, add 2px margin for interpolation kernel
        let x = ((output.x as f64 * sx).floor() as u32).saturating_sub(2);
        let y = ((output.y as f64 * sy).floor() as u32).saturating_sub(2);
        let x2 = (((output.x + output.width) as f64 * sx).ceil() as u32 + 2)
            .min(self.source_info.width);
        let y2 = (((output.y + output.height) as f64 * sy).ceil() as u32 + 2)
            .min(self.source_info.height);
        Rect::new(x, y, x2 - x, y2 - y)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::LocalNeighborhood
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
            rasmcore_pipeline::MetadataValue::Int(self.target_width as i64),
        );
        meta.set(
            "height",
            rasmcore_pipeline::MetadataValue::Int(self.target_height as i64),
        );
        Some(meta)
    }
}
