//! WIT adapter for the pipeline resource.

use std::cell::RefCell;

use crate::bindings::exports::rasmcore::image::pipeline::{
    self, ExifOrientation, FlipDirection, GuestImagePipeline, NodeId, ResizeFilter, Rotation,
};
use crate::bindings::rasmcore::core::{errors::RasmcoreError, types};

use crate::domain;
use crate::domain::pipeline::graph::NodeGraph;
use crate::domain::pipeline::nodes::{color, composite, filters, sink, source, transform};

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

fn to_domain_tiff_compression_pipeline(
    c: Option<pipeline::TiffCompression>,
) -> crate::domain::encoder::tiff::TiffCompression {
    match c {
        None => crate::domain::encoder::tiff::TiffCompression::Lzw,
        Some(pipeline::TiffCompression::None) => {
            crate::domain::encoder::tiff::TiffCompression::None
        }
        Some(pipeline::TiffCompression::Lzw) => crate::domain::encoder::tiff::TiffCompression::Lzw,
        Some(pipeline::TiffCompression::Deflate) => {
            crate::domain::encoder::tiff::TiffCompression::Deflate
        }
        Some(pipeline::TiffCompression::Packbits) => {
            crate::domain::encoder::tiff::TiffCompression::PackBits
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

    fn icc_to_srgb(&self, source: NodeId, icc_profile: Vec<u8>) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node =
            color::IccToSrgbNode::new(source, src_info, icc_profile).map_err(to_wit_error)?;
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn auto_orient(
        &self,
        source: NodeId,
        orientation: ExifOrientation,
    ) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let domain_orient = match orientation {
            ExifOrientation::Normal => domain::metadata::ExifOrientation::Normal,
            ExifOrientation::FlipHorizontal => domain::metadata::ExifOrientation::FlipHorizontal,
            ExifOrientation::Rotate180 => domain::metadata::ExifOrientation::Rotate180,
            ExifOrientation::FlipVertical => domain::metadata::ExifOrientation::FlipVertical,
            ExifOrientation::Transpose => domain::metadata::ExifOrientation::Transpose,
            ExifOrientation::Rotate90 => domain::metadata::ExifOrientation::Rotate90,
            ExifOrientation::Transverse => domain::metadata::ExifOrientation::Transverse,
            ExifOrientation::Rotate270 => domain::metadata::ExifOrientation::Rotate270,
        };
        let node = transform::AutoOrientNode::new(source, src_info, domain_orient);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
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

    fn convolve(
        &self,
        source: NodeId,
        kernel: Vec<f32>,
        kernel_width: u32,
        kernel_height: u32,
        divisor: f32,
    ) -> Result<NodeId, RasmcoreError> {
        let src_info = self.graph.borrow().node_info(source).map_err(to_wit_error)?;
        let node = filters::ConvolveNode::new(
            source,
            src_info,
            kernel,
            kernel_width as usize,
            kernel_height as usize,
            divisor,
        );
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn median(&self, source: NodeId, radius: u32) -> Result<NodeId, RasmcoreError> {
        let src_info = self.graph.borrow().node_info(source).map_err(to_wit_error)?;
        let node = filters::MedianNode::new(source, src_info, radius);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn sobel(&self, source: NodeId) -> Result<NodeId, RasmcoreError> {
        let src_info = self.graph.borrow().node_info(source).map_err(to_wit_error)?;
        let node = filters::SobelNode::new(source, src_info);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn canny(
        &self,
        source: NodeId,
        low_threshold: f32,
        high_threshold: f32,
    ) -> Result<NodeId, RasmcoreError> {
        let src_info = self.graph.borrow().node_info(source).map_err(to_wit_error)?;
        let node = filters::CannyNode::new(source, src_info, low_threshold, high_threshold);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn composite(&self, fg: NodeId, bg: NodeId, x: i32, y: i32) -> Result<NodeId, RasmcoreError> {
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
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::jpeg::JpegEncodeConfig {
            quality: config.quality.unwrap_or(85),
            progressive: config.progressive.unwrap_or(false),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_jpeg(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_png(
        &self,
        source: NodeId,
        config: pipeline::PngWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::png::PngEncodeConfig {
            compression_level: config.compression_level.unwrap_or(6),
            filter_type: to_domain_png_filter_pipeline(config.filter_type),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_png(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_webp(
        &self,
        source: NodeId,
        config: pipeline::WebpWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::webp::WebpEncodeConfig {
            quality: config.quality.unwrap_or(75),
            lossless: config.lossless.unwrap_or(false),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_webp(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_bmp(
        &self,
        source: NodeId,
        _config: pipeline::BmpWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        sink::write_bmp(&mut self.graph.borrow_mut(), source).map_err(to_wit_error)
    }

    fn write_ico(
        &self,
        source: NodeId,
        _config: pipeline::IcoWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        sink::write_ico(&mut self.graph.borrow_mut(), source).map_err(to_wit_error)
    }

    fn write_qoi(
        &self,
        source: NodeId,
        _config: pipeline::QoiWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        sink::write_qoi(&mut self.graph.borrow_mut(), source).map_err(to_wit_error)
    }

    fn write_gif(
        &self,
        source: NodeId,
        config: pipeline::GifWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::gif::GifEncodeConfig {
            repeat: config.repeat.unwrap_or(0),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_gif(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_avif(
        &self,
        source: NodeId,
        config: pipeline::AvifWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::avif::AvifEncodeConfig {
            quality: config.quality.unwrap_or(75),
            speed: config.speed.unwrap_or(6),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_avif(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_tiff(
        &self,
        source: NodeId,
        config: pipeline::TiffWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::tiff::TiffEncodeConfig {
            compression: to_domain_tiff_compression_pipeline(config.compression),
        };
        sink::write_tiff(&mut self.graph.borrow_mut(), source, &cfg).map_err(to_wit_error)
    }

    fn write(
        &self,
        source: NodeId,
        format: String,
        quality: Option<u8>,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write(
            &mut self.graph.borrow_mut(),
            source,
            &format,
            quality,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }
}
