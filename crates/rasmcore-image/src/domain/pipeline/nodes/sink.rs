//! Sink nodes — drive execution by pulling all regions and encoding output.

use crate::domain::encoder;
use crate::domain::error::ImageError;
use crate::domain::metadata_set::MetadataSet;
use crate::domain::pipeline::graph::NodeGraph;
use rasmcore_pipeline::Rect;

/// Embed metadata into encoded output bytes.
///
/// Applies metadata embedding in order: EXIF, XMP, IPTC, ICC.
/// Each embedding inserts marker segments after SOI (JPEG) or chunks after IHDR (PNG).
fn embed_metadata(
    encoded: Vec<u8>,
    format: &str,
    metadata: &MetadataSet,
) -> Result<Vec<u8>, ImageError> {
    let mut result = encoded;

    match format {
        "jpeg" => {
            // JPEG: embed in order — EXIF (APP1), XMP (APP1), IPTC (APP13), ICC (APP2)
            if let Some(ref exif) = metadata.exif {
                result = encoder::jpeg::embed_exif(&result, exif)?;
            }
            if let Some(ref xmp) = metadata.xmp {
                result = encoder::jpeg::embed_xmp(&result, xmp)?;
            }
            if let Some(ref iptc) = metadata.iptc {
                result = encoder::jpeg::embed_iptc(&result, iptc)?;
            }
            if let Some(ref icc) = metadata.icc_profile {
                result = encoder::jpeg::embed_icc_profile(&result, icc)?;
            }
        }
        "png" => {
            // PNG: embed EXIF (eXIf chunk) and ICC (iCCP chunk) after IHDR
            if let Some(ref exif) = metadata.exif {
                result = encoder::png::embed_exif(&result, exif)?;
            }
            if let Some(ref icc) = metadata.icc_profile {
                result = encoder::png::embed_icc_profile(&result, icc)?;
            }
            // PNG text chunks from format_specific are deferred to a future enhancement
        }
        _ => {
            // Other formats: metadata embedding not yet supported, pass through
        }
    }

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

/// Write a node's output as AVIF with typed config and optional metadata.
pub fn write_avif(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::avif::AvifEncodeConfig,
    metadata: Option<&MetadataSet>,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let encoded = encoder::avif::encode(&pixels, &info, config)?;

    match metadata {
        Some(ms) if !ms.is_empty() => embed_metadata(encoded, "avif", ms),
        _ => Ok(encoded),
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

/// Write a node's output as TIFF with typed config.
pub fn write_tiff(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::tiff::TiffEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    encoder::tiff::encode(&pixels, &info, config)
}

/// Write a node's output as BMP.
pub fn write_bmp(graph: &mut NodeGraph, node_id: u32) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    encoder::bmp::encode(&img, &info, &encoder::bmp::BmpEncodeConfig)
}

/// Write a node's output as ICO.
pub fn write_ico(graph: &mut NodeGraph, node_id: u32) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    encoder::ico::encode(&img, &info, &encoder::ico::IcoEncodeConfig)
}

/// Write a node's output as QOI.
pub fn write_qoi(graph: &mut NodeGraph, node_id: u32) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = graph.request_region(node_id, full)?;
    let img = encoder::pixels_to_dynamic_image(&pixels, &info)?;
    encoder::qoi::encode(&img, &info, &encoder::qoi::QoiEncodeConfig)
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
