//! Sink nodes — drive execution by pulling all regions and encoding output.

use crate::domain::encoder;
use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::NodeGraph;
use crate::domain::pipeline::rect::Rect;

/// Write a node's output as the given format. Drives the entire pipeline.
pub fn write(
    graph: &mut NodeGraph,
    node_id: u32,
    format: &str,
    quality: Option<u8>,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);

    // For v1, materialize the full image and encode.
    // Future: iterate in chunks matching encoder requirements.
    let pixels = graph.request_region(node_id, full)?;
    encoder::encode(&pixels, &info, format, quality)
}

/// Write a node's output as JPEG with typed config.
pub fn write_jpeg(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::jpeg::JpegEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    encoder::jpeg::encode(&img, &info, config)
}

/// Write a node's output as PNG with typed config.
pub fn write_png(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::png::PngEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    encoder::png::encode(&img, &info, config)
}

/// Write a node's output as WebP with typed config.
pub fn write_webp(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::webp::WebpEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    encoder::webp::encode(&img, &info, config)
}
