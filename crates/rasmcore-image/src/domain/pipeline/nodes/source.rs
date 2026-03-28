//! Source node — wraps the decoder. Lazy: decodes on first region request.

use std::cell::RefCell;

use crate::domain::decoder;
use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode, bytes_per_pixel};
use crate::domain::pipeline::rect::{Overlap, Rect};
use crate::domain::types::{DecodedImage, ImageInfo};

/// Source node that decodes an image lazily on first region request.
pub struct SourceNode {
    encoded_data: Vec<u8>,
    info: ImageInfo,
    decoded: RefCell<Option<DecodedImage>>,
}

impl SourceNode {
    pub fn new(data: Vec<u8>) -> Result<Self, ImageError> {
        let decoded = decoder::decode(&data)?;
        let info = decoded.info.clone();
        Ok(Self {
            encoded_data: data,
            info,
            decoded: RefCell::new(Some(decoded)),
        })
    }

    fn ensure_decoded(&self) -> Result<(), ImageError> {
        if self.decoded.borrow().is_none() {
            let decoded = decoder::decode(&self.encoded_data)?;
            *self.decoded.borrow_mut() = Some(decoded);
        }
        Ok(())
    }
}

impl ImageNode for SourceNode {
    fn info(&self) -> ImageInfo {
        self.info.clone()
    }

    fn compute_region(
        &self,
        request: Rect,
        _upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        self.ensure_decoded()?;
        let decoded = self.decoded.borrow();
        let decoded = decoded.as_ref().unwrap();

        let bpp = bytes_per_pixel(decoded.info.format) as usize;
        let src_stride = decoded.info.width as usize * bpp;
        let dst_stride = request.width as usize * bpp;
        let mut result = Vec::with_capacity(request.height as usize * dst_stride);

        for row in 0..request.height as usize {
            let src_y = request.y as usize + row;
            if src_y >= decoded.info.height as usize {
                break;
            }
            let src_start = src_y * src_stride + request.x as usize * bpp;
            let src_end = (src_start + dst_stride).min(decoded.pixels.len());
            result.extend_from_slice(&decoded.pixels[src_start..src_end]);
        }

        Ok(result)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}
