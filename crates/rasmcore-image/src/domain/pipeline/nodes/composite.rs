//! Composite node — Porter-Duff "over" blend of two upstream sources.

use crate::domain::composite;
use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::types::*;
use rasmcore_pipeline::{Overlap, Rect};

/// Composites foreground over background with an (x, y) offset.
///
/// Takes two upstream node-ids: foreground and background.
/// Output dimensions match the background.
pub struct CompositeNode {
    fg_upstream: u32,
    bg_upstream: u32,
    fg_info: ImageInfo,
    bg_info: ImageInfo,
    offset_x: i32,
    offset_y: i32,
}

impl CompositeNode {
    pub fn new(
        fg_upstream: u32,
        bg_upstream: u32,
        fg_info: ImageInfo,
        bg_info: ImageInfo,
        offset_x: i32,
        offset_y: i32,
    ) -> Self {
        Self {
            fg_upstream,
            bg_upstream,
            fg_info,
            bg_info,
            offset_x,
            offset_y,
        }
    }
}

impl ImageNode for CompositeNode {
    fn info(&self) -> ImageInfo {
        self.bg_info.clone()
    }

    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let fg_full = Rect::new(0, 0, self.fg_info.width, self.fg_info.height);
        let bg_full = Rect::new(0, 0, self.bg_info.width, self.bg_info.height);

        let fg_pixels = upstream_fn(self.fg_upstream, fg_full)?;
        let bg_pixels = upstream_fn(self.bg_upstream, bg_full)?;

        composite::alpha_composite_over(
            &fg_pixels,
            &self.fg_info,
            &bg_pixels,
            &self.bg_info,
            self.offset_x,
            self.offset_y,
        )
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}
