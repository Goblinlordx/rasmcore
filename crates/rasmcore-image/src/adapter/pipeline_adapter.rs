//! WIT adapter for the pipeline resource.

use std::cell::RefCell;

use crate::bindings::exports::rasmcore::image::pipeline::{
    self, FlipDirection, GuestImagePipeline, NodeId, ResizeFilter, Rotation,
};
use crate::bindings::rasmcore::core::{errors::RasmcoreError, types};

use crate::domain;
use crate::domain::pipeline::graph::NodeGraph;
use crate::domain::pipeline::nodes::{composite, filters, sink, source, transform};

use super::to_wit_error;
use super::to_wit_image_info;

fn to_domain_png_filter_pipeline(
    f: Option<pipeline::PngFilterType>,
) -> crate::domain::encoder::png::PngFilterType {
    match f {
        None => crate::domain::encoder::png::PngFilterType::Adaptive,
        Some(pipeline::PngFilterType::NoFilter) => {
            crate::domain::encoder::png::PngFilterType::NoFilter
        }
        Some(pipeline::PngFilterType::Sub) => crate::domain::encoder::png::PngFilterType::Sub,
        Some(pipeline::PngFilterType::Up) => crate::domain::encoder::png::PngFilterType::Up,
        Some(pipeline::PngFilterType::Avg) => crate::domain::encoder::png::PngFilterType::Avg,
        Some(pipeline::PngFilterType::Paeth) => crate::domain::encoder::png::PngFilterType::Paeth,
        Some(pipeline::PngFilterType::Adaptive) => {
            crate::domain::encoder::png::PngFilterType::Adaptive
        }
    }
}

/// Pipeline resource implementation wrapping the domain NodeGraph.
pub struct PipelineResource {
    graph: RefCell<NodeGraph>,
}

impl GuestImagePipeline for PipelineResource {
    fn new() -> Self {
        Self {
            graph: RefCell::new(NodeGraph::new(16 * 1024 * 1024)), // 16MB cache budget
        }
    }

    fn read(&self, data: Vec<u8>) -> Result<NodeId, RasmcoreError> {
        let node = source::SourceNode::new(data).map_err(to_wit_error)?;
        let id = self.graph.borrow_mut().add_node(Box::new(node));
        Ok(id)
    }

    fn node_info(&self, node: NodeId) -> Result<types::ImageInfo, RasmcoreError> {
        let info = self.graph.borrow().node_info(node).map_err(to_wit_error)?;
        Ok(to_wit_image_info(&info))
    }

    fn resize(
        &self,
        source: NodeId,
        width: u32,
        height: u32,
        filter: ResizeFilter,
    ) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let domain_filter = match filter {
            ResizeFilter::Nearest => domain::types::ResizeFilter::Nearest,
            ResizeFilter::Bilinear => domain::types::ResizeFilter::Bilinear,
            ResizeFilter::Bicubic => domain::types::ResizeFilter::Bicubic,
            ResizeFilter::Lanczos3 => domain::types::ResizeFilter::Lanczos3,
        };
        let node = transform::ResizeNode::new(source, src_info, width, height, domain_filter);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn crop(
        &self,
        source: NodeId,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node = transform::CropNode::new(source, src_info, x, y, width, height);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn rotate(&self, source: NodeId, angle: Rotation) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let domain_rot = match angle {
            Rotation::R90 => domain::types::Rotation::R90,
            Rotation::R180 => domain::types::Rotation::R180,
            Rotation::R270 => domain::types::Rotation::R270,
        };
        let node = transform::RotateNode::new(source, src_info, domain_rot);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn flip(&self, source: NodeId, direction: FlipDirection) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let domain_dir = match direction {
            FlipDirection::Horizontal => domain::types::FlipDirection::Horizontal,
            FlipDirection::Vertical => domain::types::FlipDirection::Vertical,
        };
        let node = transform::FlipNode::new(source, src_info, domain_dir);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn convert_format(
        &self,
        _source: NodeId,
        _target: types::PixelFormat,
    ) -> Result<NodeId, RasmcoreError> {
        Err(RasmcoreError::NotImplemented)
    }

    fn blur(&self, source: NodeId, radius: f32) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node = filters::BlurNode::new(source, src_info, radius);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn sharpen(&self, source: NodeId, amount: f32) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node = filters::SharpenNode::new(source, src_info, amount);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn brightness(&self, source: NodeId, amount: f32) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node = filters::BrightnessNode::new(source, src_info, amount);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn contrast(&self, source: NodeId, amount: f32) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node = filters::ContrastNode::new(source, src_info, amount);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn grayscale(&self, source: NodeId) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node = filters::GrayscaleNode::new(source, src_info);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn composite(
        &self,
        fg: NodeId,
        bg: NodeId,
        x: i32,
        y: i32,
    ) -> Result<NodeId, RasmcoreError> {
        let graph = self.graph.borrow();
        let fg_info = graph.node_info(fg).map_err(to_wit_error)?;
        let bg_info = graph.node_info(bg).map_err(to_wit_error)?;
        drop(graph);
        let node = composite::CompositeNode::new(fg, bg, fg_info, bg_info, x, y);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn write_jpeg(
        &self,
        source: NodeId,
        config: pipeline::JpegWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::jpeg::JpegEncodeConfig {
            quality: config.quality.unwrap_or(85),
        };
        sink::write_jpeg(&mut self.graph.borrow_mut(), source, &cfg).map_err(to_wit_error)
    }

    fn write_png(
        &self,
        source: NodeId,
        config: pipeline::PngWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::png::PngEncodeConfig {
            compression_level: config.compression_level.unwrap_or(6),
            filter_type: to_domain_png_filter_pipeline(config.filter_type),
        };
        sink::write_png(&mut self.graph.borrow_mut(), source, &cfg).map_err(to_wit_error)
    }

    fn write_webp(
        &self,
        source: NodeId,
        config: pipeline::WebpWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::webp::WebpEncodeConfig {
            quality: config.quality.unwrap_or(75),
            lossless: config.lossless.unwrap_or(false),
        };
        sink::write_webp(&mut self.graph.borrow_mut(), source, &cfg).map_err(to_wit_error)
    }

    fn write(
        &self,
        source: NodeId,
        format: String,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        sink::write(&mut self.graph.borrow_mut(), source, &format, quality).map_err(to_wit_error)
    }
}
