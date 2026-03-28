//! Transform nodes — wrap existing domain transform operations.

use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform;
use crate::domain::types::*;
use rasmcore_pipeline::{Overlap, Rect};

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

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::LocalNeighborhood
    }
}

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

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

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

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }
}

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

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }
}
