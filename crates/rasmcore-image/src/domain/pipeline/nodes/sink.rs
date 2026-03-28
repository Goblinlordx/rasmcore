//! Sink nodes — drive execution by pulling all regions and encoding output.

use crate::domain::encoder;
use crate::domain::error::ImageError;
use crate::domain::metadata_set::MetadataSet;
use crate::domain::pipeline::graph::NodeGraph;
use rasmcore_pipeline::Rect;

/// Embed metadata into encoded output bytes.
/// Currently supports ICC profile embedding for JPEG and PNG.
fn embed_metadata(
    encoded: Vec<u8>,
    format: &str,
    metadata: &MetadataSet,
) -> Result<Vec<u8>, ImageError> {
    let mut result = encoded;

    // Embed ICC profile if present
    if let Some(ref icc) = metadata.icc_profile {
        result = match format {
            "jpeg" => encoder::jpeg::embed_icc_profile(&result, icc)?,
            "png" => encoder::png::embed_icc_profile(&result, icc)?,
            _ => result, // Other formats: ICC embedding not yet supported
        };
    }

    // EXIF, XMP, IPTC embedding will be added in the metadata-formats track
    Ok(result)
}

/// Write a node's output as the given format. Drives the entire pipeline.
pub fn write(
    graph: &mut NodeGraph,
    node_id: u32,
    format: &str,
    quality: Option<u8>,
    metadata: Option<&MetadataSet>,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let encoded = encoder::encode(&pixels, &info, format, quality)?;

    match metadata {
        Some(ms) => embed_metadata(encoded, format, ms),
        None => Ok(encoded),
    }
}

/// Write a node's output as JPEG with typed config and optional metadata.
pub fn write_jpeg(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::jpeg::JpegEncodeConfig,
    metadata: Option<&MetadataSet>,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let encoded = encoder::jpeg::encode_pixels(&pixels, &info, config)?;

    match metadata {
        Some(ms) => embed_metadata(encoded, "jpeg", ms),
        None => Ok(encoded),
    }
}

/// Write a node's output as PNG with typed config and optional metadata.
pub fn write_png(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::png::PngEncodeConfig,
    metadata: Option<&MetadataSet>,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    let encoded = encoder::png::encode(&img, &info, config)?;

    match metadata {
        Some(ms) => embed_metadata(encoded, "png", ms),
        None => Ok(encoded),
    }
}

/// Write a node's output as GIF with typed config and optional metadata.
pub fn write_gif(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::gif::GifEncodeConfig,
    metadata: Option<&MetadataSet>,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    let encoded = encoder::gif::encode(&img, &info, config)?;

    match metadata {
        Some(ms) if !ms.is_empty() => embed_metadata(encoded, "gif", ms),
        _ => Ok(encoded),
    }
}

/// Write a node's output as WebP with typed config and optional metadata.
pub fn write_webp(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::webp::WebpEncodeConfig,
    metadata: Option<&MetadataSet>,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    let encoded = encoder::webp::encode(&img, &info, config)?;

    match metadata {
        Some(ms) if !ms.is_empty() => embed_metadata(encoded, "webp", ms),
        _ => Ok(encoded),
    }
}
