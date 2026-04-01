//! Composite node — blends two upstream sources with optional blend mode.

use crate::domain::composite;
use crate::domain::error::ImageError;
use crate::domain::filters::BlendMode;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

/// Composites foreground over background with an (x, y) offset.
///
/// Takes two upstream node-ids: foreground and background.
/// Output dimensions match the background.
/// When `blend_mode` is None, uses Porter-Duff "over" (alpha composite).
/// When `blend_mode` is Some, uses the specified blend mode.
pub struct CompositeNode {
    fg_upstream: u32,
    bg_upstream: u32,
    fg_info: ImageInfo,
    bg_info: ImageInfo,
    offset_x: i32,
    offset_y: i32,
    blend_mode: Option<BlendMode>,
}

impl CompositeNode {
    pub fn new(
        fg_upstream: u32,
        bg_upstream: u32,
        fg_info: ImageInfo,
        bg_info: ImageInfo,
        offset_x: i32,
        offset_y: i32,
        blend_mode: Option<BlendMode>,
    ) -> Self {
        Self {
            fg_upstream,
            bg_upstream,
            fg_info,
            bg_info,
            offset_x,
            offset_y,
            blend_mode,
        }
    }
}

#[rasmcore_macros::register_transform(name = "composite")]
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

        match self.blend_mode {
            None => composite::alpha_composite_over(
                &fg_pixels,
                &self.fg_info,
                &bg_pixels,
                &self.bg_info,
                self.offset_x,
                self.offset_y,
            ),
            Some(ref mode) => {
                // For blend modes, we first composite (position fg on bg-sized canvas),
                // then blend. This handles different-size images with offset.
                let positioned = composite::alpha_composite_over(
                    &fg_pixels,
                    &self.fg_info,
                    // Transparent background matching bg dimensions
                    &vec![0u8; self.bg_info.width as usize * self.bg_info.height as usize * 4],
                    &ImageInfo {
                        width: self.bg_info.width,
                        height: self.bg_info.height,
                        format: PixelFormat::Rgba8,
                        color_space: self.bg_info.color_space,
                    },
                    self.offset_x,
                    self.offset_y,
                )?;
                let positioned_info = ImageInfo {
                    width: self.bg_info.width,
                    height: self.bg_info.height,
                    format: PixelFormat::Rgba8,
                    color_space: self.bg_info.color_space,
                };
                crate::domain::filters::blend(
                    &positioned,
                    &positioned_info,
                    &bg_pixels,
                    &self.bg_info,
                    *mode,
                )
            }
        }
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}
