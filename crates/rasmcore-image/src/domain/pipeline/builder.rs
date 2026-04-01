//! Pipeline builder — validated wrapper around NodeGraph + GraphDescription.
//!
//! The builder provides the same node-creation API as `NodeGraph` but:
//! 1. Validates parameters at build time (upstream refs, dimensions, bounds)
//! 2. Builds a `GraphDescription` alongside the live `NodeGraph`
//! 3. Can later support description-only building (no live graph)

use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{GraphDescription, NodeGraph, NodeKind};
use crate::domain::types::ImageInfo;

/// Validated pipeline builder that wraps `NodeGraph` and `GraphDescription`.
///
/// Every node-creation method validates parameters before delegating to the
/// underlying `NodeGraph`. The `GraphDescription` is built in parallel and
/// can be extracted for serialization, introspection, or later execution.
pub struct PipelineBuilder {
    graph: NodeGraph,
}

impl PipelineBuilder {
    /// Create a new builder with the given spatial cache budget.
    pub fn new(cache_budget: usize) -> Self {
        Self {
            graph: NodeGraph::new(cache_budget),
        }
    }

    /// Create a builder wrapping an existing `NodeGraph`.
    pub fn from_graph(graph: NodeGraph) -> Self {
        Self { graph }
    }

    /// Borrow the underlying `NodeGraph` (read-only).
    pub fn graph(&self) -> &NodeGraph {
        &self.graph
    }

    /// Borrow the underlying `NodeGraph` (mutable — for execution).
    pub fn graph_mut(&mut self) -> &mut NodeGraph {
        &mut self.graph
    }

    /// Consume the builder and return the underlying `NodeGraph`.
    pub fn into_graph(self) -> NodeGraph {
        self.graph
    }

    /// Get the graph description built alongside the live graph.
    pub fn description(&self) -> &GraphDescription {
        self.graph.description()
    }

    /// Validate an upstream reference exists in the graph.
    pub fn validate_upstream(&self, upstream_id: u32) -> Result<(), ImageError> {
        self.graph.validate_upstream(upstream_id).map_err(Into::into)
    }

    /// Validate the full graph before execution.
    pub fn validate(&self) -> Result<(), ImageError> {
        self.graph.validate().map_err(Into::into)
    }

    /// Get image info for a node.
    pub fn node_info(&self, node_id: u32) -> Result<ImageInfo, ImageError> {
        self.graph.node_info(node_id)
    }

    // ─── Validated node creation ──────────────────────────────────────────

    /// Add a source node (no upstream, no validation needed).
    pub fn add_source(
        &mut self,
        node: Box<dyn super::graph::ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        metadata: rasmcore_pipeline::Metadata,
    ) -> u32 {
        self.graph
            .add_node_with_hash_and_metadata(node, hash, metadata)
    }

    /// Add a filter/transform node with upstream validation.
    ///
    /// Validates that `upstream_id` exists in the graph before creating the node.
    pub fn add_with_upstream(
        &mut self,
        node: Box<dyn super::graph::ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        upstream_id: u32,
    ) -> Result<u32, ImageError> {
        self.graph
            .validate_upstream(upstream_id)
            .map_err(ImageError::from)?;
        Ok(self.graph.add_node_derived(node, hash, upstream_id))
    }

    /// Add a filter/transform node with full descriptor tracking.
    ///
    /// Validates upstream, records kind/name in the graph description.
    pub fn add_described(
        &mut self,
        node: Box<dyn super::graph::ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        upstream_id: u32,
        kind: NodeKind,
        name: &str,
    ) -> Result<u32, ImageError> {
        self.graph
            .validate_upstream(upstream_id)
            .map_err(ImageError::from)?;
        Ok(self
            .graph
            .add_node_described(node, hash, upstream_id, kind, name))
    }

    /// Validate resize parameters before creating a resize node.
    pub fn validate_resize(width: u32, height: u32) -> Result<(), ImageError> {
        super::graph::validate_resize(width, height).map_err(Into::into)
    }

    /// Validate crop parameters before creating a crop node.
    pub fn validate_crop(
        src_info: &ImageInfo,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), ImageError> {
        super::graph::validate_crop(src_info.width, src_info.height, x, y, width, height)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
    use crate::domain::types::{ColorSpace, PixelFormat};
    use rasmcore_pipeline::Rect;

    struct TestSource {
        info: ImageInfo,
    }

    impl ImageNode for TestSource {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }
        fn compute_region(
            &self,
            request: Rect,
            _: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            let bpp = crate::domain::types::bytes_per_pixel(self.info.format);
            Ok(vec![
                128;
                request.width as usize * request.height as usize * bpp as usize
            ])
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
    }

    struct TestFilter {
        upstream: u32,
        info: ImageInfo,
    }

    impl ImageNode for TestFilter {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }
        fn compute_region(
            &self,
            request: Rect,
            upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            upstream_fn(self.upstream, request)
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
        fn upstream_id(&self) -> Option<u32> {
            Some(self.upstream)
        }
    }

    fn test_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn builder_add_source_and_filter() {
        let mut b = PipelineBuilder::new(1024 * 1024);
        let src = b.add_source(
            Box::new(TestSource {
                info: test_info(100, 100),
            }),
            rasmcore_pipeline::ZERO_HASH,
            rasmcore_pipeline::Metadata::new(),
        );
        assert_eq!(src, 0);

        let filter = b
            .add_with_upstream(
                Box::new(TestFilter {
                    upstream: src,
                    info: test_info(100, 100),
                }),
                rasmcore_pipeline::ZERO_HASH,
                src,
            )
            .unwrap();
        assert_eq!(filter, 1);
        assert!(b.validate().is_ok());
    }

    #[test]
    fn builder_rejects_invalid_upstream() {
        let mut b = PipelineBuilder::new(1024 * 1024);
        let result = b.add_with_upstream(
            Box::new(TestFilter {
                upstream: 99,
                info: test_info(100, 100),
            }),
            rasmcore_pipeline::ZERO_HASH,
            99, // doesn't exist
        );
        assert!(result.is_err());
    }

    #[test]
    fn builder_description_tracks_nodes() {
        let mut b = PipelineBuilder::new(1024 * 1024);
        b.add_source(
            Box::new(TestSource {
                info: test_info(640, 480),
            }),
            rasmcore_pipeline::ZERO_HASH,
            rasmcore_pipeline::Metadata::new(),
        );
        let desc = b.description();
        assert_eq!(desc.len(), 1);
        assert_eq!(desc.get(0).unwrap().output_info.width, 640);
    }

    #[test]
    fn builder_validate_resize() {
        assert!(PipelineBuilder::validate_resize(100, 100).is_ok());
        assert!(PipelineBuilder::validate_resize(0, 100).is_err());
        assert!(PipelineBuilder::validate_resize(100, 0).is_err());
    }

    #[test]
    fn builder_validate_crop() {
        let info = test_info(100, 100);
        assert!(PipelineBuilder::validate_crop(&info, 10, 10, 50, 50).is_ok());
        assert!(PipelineBuilder::validate_crop(&info, 60, 60, 50, 50).is_err());
        assert!(PipelineBuilder::validate_crop(&info, 0, 0, 0, 50).is_err());
    }

    #[test]
    fn builder_described_node() {
        let mut b = PipelineBuilder::new(1024 * 1024);
        let src = b.add_source(
            Box::new(TestSource {
                info: test_info(100, 100),
            }),
            rasmcore_pipeline::ZERO_HASH,
            rasmcore_pipeline::Metadata::new(),
        );
        let filter = b
            .add_described(
                Box::new(TestFilter {
                    upstream: src,
                    info: test_info(100, 100),
                }),
                rasmcore_pipeline::ZERO_HASH,
                src,
                NodeKind::Filter,
                "blur",
            )
            .unwrap();

        let desc = b.description();
        assert_eq!(desc.len(), 2);
        let d1 = desc.get(filter).unwrap();
        assert_eq!(d1.kind, NodeKind::Filter);
        assert_eq!(d1.name, "blur");
    }

    #[test]
    fn builder_validate_empty() {
        let b = PipelineBuilder::new(1024 * 1024);
        assert!(b.validate().is_err());
    }
}
