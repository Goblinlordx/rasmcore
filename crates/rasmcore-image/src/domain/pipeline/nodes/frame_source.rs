//! Frame-aware source node — wraps multi-frame decode with frame selection.
//!
//! The FrameSourceNode holds raw image data and a FrameSelection. On each
//! pipeline execution pass, it serves pixels from the current frame. The
//! executor advances the frame index between passes via `set_current_frame()`.

use std::cell::RefCell;

use crate::domain::decoder;
use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode, bytes_per_pixel};
use crate::domain::types::{DecodedImage, FrameInfo, FrameSelection, ImageInfo};
use rasmcore_pipeline::{Overlap, Rect};

/// Source node for multi-frame images (GIF, WebP, TIFF).
///
/// Holds the raw encoded data and decodes individual frames on demand.
/// The executor calls `set_current_frame()` to advance between frames,
/// then runs the graph for each selected frame.
pub struct FrameSourceNode {
    encoded_data: Vec<u8>,
    selection: FrameSelection,
    total_frames: u32,
    /// The currently active frame (decoded on demand).
    current: RefCell<Option<(DecodedImage, FrameInfo)>>,
    /// Info for the current frame (updated when frame changes).
    current_info: RefCell<ImageInfo>,
    /// Canvas dimensions (from the first frame / logical screen).
    canvas_width: u32,
    canvas_height: u32,
}

impl FrameSourceNode {
    /// Create a new frame source. Peeks at the first selected frame to
    /// determine canvas dimensions and initial image info.
    pub fn new(data: Vec<u8>, selection: FrameSelection) -> Result<Self, ImageError> {
        let total_frames = decoder::frame_count(&data)?;
        let first_idx = match &selection {
            FrameSelection::Single(i) => *i,
            FrameSelection::Pick(v) => *v.first().ok_or_else(|| {
                ImageError::InvalidParameters("Pick selection is empty".into())
            })?,
            FrameSelection::Range(start, _) => *start,
            FrameSelection::All => 0,
        };

        let (decoded, frame_info) = decoder::decode_frame(&data, first_idx)?;
        let info = decoded.info.clone();
        let canvas_width = info.width;
        let canvas_height = info.height;

        Ok(Self {
            encoded_data: data,
            selection,
            total_frames,
            current: RefCell::new(Some((decoded, frame_info))),
            current_info: RefCell::new(info),
            canvas_width,
            canvas_height,
        })
    }

    /// Resolve the list of frame indices from the selection.
    pub fn selected_indices(&self) -> Vec<u32> {
        match &self.selection {
            FrameSelection::Single(i) => vec![*i],
            FrameSelection::Pick(v) => v.clone(),
            FrameSelection::Range(start, end) => (*start..*end).collect(),
            FrameSelection::All => (0..self.total_frames).collect(),
        }
    }

    /// Total number of frames in the source (not just selected).
    pub fn total_frames(&self) -> u32 {
        self.total_frames
    }

    /// Canvas dimensions.
    pub fn canvas_size(&self) -> (u32, u32) {
        (self.canvas_width, self.canvas_height)
    }

    /// Set the current frame to decode. Called by the executor between passes.
    pub fn set_current_frame(&self, index: u32) -> Result<FrameInfo, ImageError> {
        let (decoded, frame_info) = decoder::decode_frame(&self.encoded_data, index)?;
        *self.current_info.borrow_mut() = decoded.info.clone();
        *self.current.borrow_mut() = Some((decoded, frame_info.clone()));
        Ok(frame_info)
    }

    /// Get the FrameInfo for the current frame (if loaded).
    pub fn current_frame_info(&self) -> Option<FrameInfo> {
        self.current.borrow().as_ref().map(|(_, fi)| fi.clone())
    }
}

impl ImageNode for FrameSourceNode {
    fn info(&self) -> ImageInfo {
        self.current_info.borrow().clone()
    }

    fn compute_region(
        &self,
        request: Rect,
        _upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let current = self.current.borrow();
        let (decoded, _) = current.as_ref().ok_or_else(|| {
            ImageError::InvalidInput("FrameSourceNode: no frame loaded".into())
        })?;

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
