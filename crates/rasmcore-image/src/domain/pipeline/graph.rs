//! Node graph — holds all nodes and dispatches region requests.

use crate::domain::error::ImageError;
use crate::domain::pipeline::nodes::frame_source::FrameSourceNode;
use crate::domain::types::{DecodedImage, FrameSequence, ImageInfo};
use rasmcore_pipeline::{Overlap, Rect, SpatialCache};

/// Access pattern hint for cache optimization.
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    /// Output tiles map to same-position upstream (point ops, crop).
    Sequential,
    /// Output tiles need upstream with overlap (blur, resize).
    LocalNeighborhood,
    /// Output tiles may request any upstream region (rotation).
    RandomAccess,
    /// Must see all upstream tiles before producing output (histogram eq).
    GlobalTwoPass,
}

/// Blanket downcast support for pipeline nodes.
pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The core trait every pipeline node implements.
pub trait ImageNode: AsAny {
    /// Image dimensions and format (available without computing pixels).
    fn info(&self) -> ImageInfo;

    /// Compute pixels for the requested region.
    /// Uses `upstream_fn` to request regions from upstream nodes.
    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError>;

    /// Overlap this node needs from upstream for each output region.
    fn overlap(&self) -> Overlap;

    /// Access pattern hint.
    fn access_pattern(&self) -> AccessPattern;
}

/// Pipeline-owned node graph.
pub struct NodeGraph {
    nodes: Vec<Box<dyn ImageNode>>,
    cache: SpatialCache,
    cache_budget: usize,
}

impl NodeGraph {
    /// Create a new graph with the given memory budget for the spatial cache.
    pub fn new(cache_budget: usize) -> Self {
        Self {
            nodes: Vec::new(),
            cache: SpatialCache::new(cache_budget),
            cache_budget,
        }
    }

    /// Add a node to the graph. Returns its node-id.
    pub fn add_node(&mut self, node: Box<dyn ImageNode>) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        id
    }

    /// Get image info for a node (no pixel computation).
    pub fn node_info(&self, node_id: u32) -> Result<ImageInfo, ImageError> {
        self.nodes
            .get(node_id as usize)
            .map(|n| n.info())
            .ok_or_else(|| ImageError::InvalidParameters(format!("invalid node id: {node_id}")))
    }

    /// Request a region from a node. Uses the spatial cache for reuse.
    pub fn request_region(&mut self, node_id: u32, request: Rect) -> Result<Vec<u8>, ImageError> {
        let info = self.node_info(node_id)?;
        let bpp = bytes_per_pixel(info.format);

        // Check cache first
        let query = self.cache.query(node_id, request);
        if query.fully_cached {
            let handle = query.hits[0];
            let cached_rect = self.cache.rect(handle);
            if cached_rect == request {
                return Ok(self.cache.read(handle).to_vec());
            }
            return Ok(self.cache.extract_subregion(handle, request, bpp));
        }

        // Compute the full requested region.
        // We use raw pointer to split the borrow: nodes[node_id] is read,
        // while the rest of self (cache, other nodes) can be mutated.
        let pixels = {
            let nodes_ptr = self.nodes.as_ptr();
            let node_count = self.nodes.len();
            if (node_id as usize) >= node_count {
                return Err(ImageError::InvalidParameters(format!(
                    "invalid node id: {node_id}"
                )));
            }
            let node = unsafe { &*nodes_ptr.add(node_id as usize) };

            let self_ptr = self as *mut NodeGraph;
            let mut upstream_fn = |upstream_id: u32, rect: Rect| -> Result<Vec<u8>, ImageError> {
                unsafe { &mut *self_ptr }.request_region(upstream_id, rect)
            };

            node.compute_region(request, &mut upstream_fn)?
        };

        // Store in cache
        let handle = self.cache.store(node_id, request, pixels, bpp);
        Ok(self.cache.read(handle).to_vec())
    }

    /// Access the spatial cache directly.
    pub fn cache(&self) -> &SpatialCache {
        &self.cache
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Execute the pipeline in sequence mode: run the graph once per selected
    /// frame from a FrameSourceNode, collecting results into a FrameSequence.
    ///
    /// `source_node_id` must point to a FrameSourceNode. `output_node_id` is
    /// the final node whose output is captured for each frame.
    pub fn execute_sequence(
        &mut self,
        source_node_id: u32,
        output_node_id: u32,
    ) -> Result<FrameSequence, ImageError> {
        // Get the selected frame indices from the FrameSourceNode.
        // We need to read them before mutably borrowing for request_region.
        let (indices, canvas_w, canvas_h) = {
            let node = self.nodes.get(source_node_id as usize).ok_or_else(|| {
                ImageError::InvalidParameters(format!("invalid source node id: {source_node_id}"))
            })?;
            let frame_src = node
                .as_any()
                .downcast_ref::<FrameSourceNode>()
                .ok_or_else(|| {
                    ImageError::InvalidParameters(
                        "source node is not a FrameSourceNode".into(),
                    )
                })?;
            let indices = frame_src.selected_indices();
            let (cw, ch) = frame_src.canvas_size();
            (indices, cw, ch)
        };

        let mut sequence = FrameSequence::new(canvas_w, canvas_h);

        for &frame_idx in &indices {
            // Advance the source to the next frame
            let frame_info = {
                let node = &self.nodes[source_node_id as usize];
                let frame_src = node.as_any().downcast_ref::<FrameSourceNode>().unwrap();
                frame_src.set_current_frame(frame_idx)?
            };

            // Clear cache between frames — cached regions from previous frame are stale
            self.cache = SpatialCache::new(self.cache_budget);

            // Execute the graph for this frame
            let info = self.node_info(output_node_id)?;
            let full = Rect::new(0, 0, info.width, info.height);
            let pixels = self.request_region(output_node_id, full)?;

            sequence.push(
                DecodedImage {
                    pixels,
                    info,
                    icc_profile: None,
                },
                frame_info,
            );
        }

        Ok(sequence)
    }
}

/// Calculate bytes per pixel for a given format.
/// Extract a sub-rectangle from a pixel buffer.
///
/// Given `src_pixels` covering `src_rect`, extracts the pixels for `sub_rect`.
/// Both rects must be in the same coordinate space, and `sub_rect` must be
/// contained within `src_rect`. Used by tiled node implementations to crop
/// the overlap padding after applying a filter.
pub fn crop_region(src_pixels: &[u8], src_rect: Rect, sub_rect: Rect, bpp: u32) -> Vec<u8> {
    debug_assert!(
        src_rect.contains(&sub_rect),
        "sub_rect {sub_rect:?} not within src_rect {src_rect:?}"
    );
    if src_rect == sub_rect {
        return src_pixels.to_vec();
    }
    let src_stride = src_rect.width as usize * bpp as usize;
    let sub_stride = sub_rect.width as usize * bpp as usize;
    let x_off = (sub_rect.x - src_rect.x) as usize * bpp as usize;
    let y_off = (sub_rect.y - src_rect.y) as usize;

    let mut out = Vec::with_capacity(sub_rect.height as usize * sub_stride);
    for row in 0..sub_rect.height as usize {
        let start = (y_off + row) * src_stride + x_off;
        out.extend_from_slice(&src_pixels[start..start + sub_stride]);
    }
    out
}

pub fn bytes_per_pixel(format: crate::domain::types::PixelFormat) -> u32 {
    use crate::domain::types::PixelFormat;
    match format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
        PixelFormat::Gray8 => 1,
        PixelFormat::Gray16 => 2,
        PixelFormat::Rgb16 => 6,
        PixelFormat::Rgba16 => 8,
        PixelFormat::Cmyk8 => 4,
        PixelFormat::Cmyka8 => 5,
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p | PixelFormat::Nv12 => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, PixelFormat};

    struct SolidColorNode {
        info: ImageInfo,
        value: u8,
    }

    impl ImageNode for SolidColorNode {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }

        fn compute_region(
            &self,
            request: Rect,
            _upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            let bpp = bytes_per_pixel(self.info.format);
            let size = request.width as usize * request.height as usize * bpp as usize;
            Ok(vec![self.value; size])
        }

        fn overlap(&self) -> Overlap {
            Overlap::zero()
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
    }

    fn make_solid(width: u32, height: u32, value: u8) -> Box<dyn ImageNode> {
        Box::new(SolidColorNode {
            info: ImageInfo {
                width,
                height,
                format: PixelFormat::Rgba8,
                color_space: ColorSpace::Srgb,
            },
            value,
        })
    }

    #[test]
    fn add_and_info() {
        let mut g = NodeGraph::new(1024 * 1024);
        let id = g.add_node(make_solid(100, 100, 0));
        assert_eq!(g.node_info(id).unwrap().width, 100);
    }

    #[test]
    fn request_region_basic() {
        let mut g = NodeGraph::new(1024 * 1024);
        let id = g.add_node(make_solid(100, 100, 42));
        let p = g.request_region(id, Rect::new(0, 0, 10, 10)).unwrap();
        assert_eq!(p.len(), 10 * 10 * 4);
        assert!(p.iter().all(|&v| v == 42));
    }

    #[test]
    fn request_cached() {
        let mut g = NodeGraph::new(1024 * 1024);
        let id = g.add_node(make_solid(100, 100, 55));
        let p1 = g.request_region(id, Rect::new(0, 0, 10, 10)).unwrap();
        let p2 = g.request_region(id, Rect::new(0, 0, 10, 10)).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn request_subregion() {
        let mut g = NodeGraph::new(1024 * 1024);
        let id = g.add_node(make_solid(100, 100, 77));
        g.request_region(id, Rect::new(0, 0, 50, 50)).unwrap();
        let sub = g.request_region(id, Rect::new(10, 10, 20, 20)).unwrap();
        assert_eq!(sub.len(), 20 * 20 * 4);
    }

    #[test]
    fn invalid_node() {
        let mut g = NodeGraph::new(1024 * 1024);
        assert!(g.request_region(99, Rect::new(0, 0, 10, 10)).is_err());
    }

    #[test]
    fn upstream_chain() {
        // Node 0: solid 100x100 value=10
        // Node 1: doubles upstream pixels (custom node)
        struct DoubleNode {
            upstream: u32,
            info: ImageInfo,
        }
        impl ImageNode for DoubleNode {
            fn info(&self) -> ImageInfo {
                self.info.clone()
            }
            fn compute_region(
                &self,
                request: Rect,
                upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
            ) -> Result<Vec<u8>, ImageError> {
                let src = upstream_fn(self.upstream, request)?;
                Ok(src.iter().map(|&v| v.saturating_mul(2)).collect())
            }
            fn overlap(&self) -> Overlap {
                Overlap::zero()
            }
            fn access_pattern(&self) -> AccessPattern {
                AccessPattern::Sequential
            }
        }

        let mut g = NodeGraph::new(1024 * 1024);
        let src = g.add_node(make_solid(100, 100, 10));
        let info = g.node_info(src).unwrap();
        let dbl = g.add_node(Box::new(DoubleNode {
            upstream: src,
            info,
        }));

        let p = g.request_region(dbl, Rect::new(0, 0, 5, 5)).unwrap();
        assert!(p.iter().all(|&v| v == 20));
    }
}

#[cfg(test)]
mod tiled_parity_tests {
    use super::*;
    use crate::domain::pipeline::nodes::filters::{
        BlurNode, BrightnessNode, ContrastNode, SharpenNode,
    };
    use crate::domain::types::*;

    /// Raw pixel source node for testing (no decode step).
    struct RawSource {
        pixels: Vec<u8>,
        info: ImageInfo,
    }
    impl RawSource {
        fn new(pixels: Vec<u8>, info: ImageInfo) -> Self {
            Self { pixels, info }
        }
    }
    impl ImageNode for RawSource {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }
        fn compute_region(
            &self,
            request: Rect,
            _: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            let bpp = bytes_per_pixel(self.info.format);
            Ok(crop_region(
                &self.pixels,
                Rect::new(0, 0, self.info.width, self.info.height),
                request,
                bpp,
            ))
        }
        fn overlap(&self) -> rasmcore_pipeline::Overlap {
            rasmcore_pipeline::Overlap::zero()
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
    }

    fn gradient_pixels(w: u32, h: u32) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push(((x * 255) / w) as u8);
                px.push(((y * 255) / h) as u8);
                px.push(128);
                px.push(255);
            }
        }
        px
    }

    fn stitch_tiles(tiles: &[(Rect, Vec<u8>)], full_w: u32, full_h: u32, bpp: u32) -> Vec<u8> {
        let stride = full_w as usize * bpp as usize;
        let mut out = vec![0u8; full_h as usize * stride];
        for (rect, pixels) in tiles {
            let tile_stride = rect.width as usize * bpp as usize;
            for row in 0..rect.height as usize {
                let dst_start = (rect.y as usize + row) * stride + rect.x as usize * bpp as usize;
                let src_start = row * tile_stride;
                out[dst_start..dst_start + tile_stride]
                    .copy_from_slice(&pixels[src_start..src_start + tile_stride]);
            }
        }
        out
    }

    /// Core validation: full-image request must produce byte-identical output
    /// to stitched tile requests. Zero tolerance.
    fn assert_tiled_matches_full(graph_builder: impl Fn() -> (NodeGraph, u32, u32, u32, u32)) {
        // Build graph and get full-image result
        let (mut g, output_node, w, h, bpp) = graph_builder();
        let full = g
            .request_region(output_node, Rect::new(0, 0, w, h))
            .unwrap();

        // Build fresh graph and request as 4 quadrant tiles
        let (mut g2, output_node2, _, _, _) = graph_builder();
        let hw = w / 2;
        let hh = h / 2;
        let tiles = vec![
            (
                Rect::new(0, 0, hw, hh),
                g2.request_region(output_node2, Rect::new(0, 0, hw, hh))
                    .unwrap(),
            ),
            (
                Rect::new(hw, 0, w - hw, hh),
                g2.request_region(output_node2, Rect::new(hw, 0, w - hw, hh))
                    .unwrap(),
            ),
            (
                Rect::new(0, hh, hw, h - hh),
                g2.request_region(output_node2, Rect::new(0, hh, hw, h - hh))
                    .unwrap(),
            ),
            (
                Rect::new(hw, hh, w - hw, h - hh),
                g2.request_region(output_node2, Rect::new(hw, hh, w - hw, h - hh))
                    .unwrap(),
            ),
        ];

        let stitched = stitch_tiles(&tiles, w, h, bpp);

        assert_eq!(
            full.len(),
            stitched.len(),
            "buffer length mismatch: full={} stitched={}",
            full.len(),
            stitched.len()
        );
        assert_eq!(
            full,
            stitched,
            "TILED OUTPUT DIFFERS FROM FULL-IMAGE OUTPUT. \
             First diff at byte {}",
            full.iter()
                .zip(stitched.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0)
        );
    }

    #[test]
    fn tiled_parity_point_op() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let bright = g.add_node(Box::new(BrightnessNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                0.2,
            )));
            (g, bright, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_blur() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let blur = g.add_node(Box::new(BlurNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                2.0,
            )));
            (g, blur, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_sharpen() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let sharp = g.add_node(Box::new(SharpenNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                1.5,
            )));
            (g, sharp, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_contrast() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let contrast = g.add_node(Box::new(ContrastNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                0.5,
            )));
            (g, contrast, w, h, 4)
        });
    }
}
