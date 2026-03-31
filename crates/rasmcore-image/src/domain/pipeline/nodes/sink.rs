//! Sink nodes — drive execution by pulling regions and encoding output.
//!
//! By default, the output image is requested as tiles (default 512×512) to
//! reduce peak memory for large images. Tiles are stitched into the full
//! output buffer before encoding, since all current encoders require
//! complete pixel buffers.
//!
//! Tile execution is pixel-identical to full-image execution — the pipeline
//! graph's `request_region` produces the same bytes regardless of request
//! size. The spatial cache automatically reuses overlapping tile edges.
//!
//! Encoder streaming capabilities (future work):
//!   - TIFF: supports strip-based writing — could accept tiles directly
//!   - PNG: scanline-based encoding possible via libpng
//!   - JPEG: scanline-based encoding possible via libjpeg
//!   - WebP, AVIF, GIF, BMP, ICO, QOI: require complete pixel buffers

use crate::domain::encoder;
use crate::domain::error::ImageError;
use crate::domain::metadata_set::MetadataSet;
use crate::domain::pipeline::graph::{bytes_per_pixel, NodeGraph};
use rasmcore_pipeline::Rect;

/// Default tile size (pixels per side). 512×512 at RGB8 = 768 KB per tile.
const DEFAULT_TILE_SIZE: u32 = 512;

/// Configuration for tiled sink execution.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Tile width and height in pixels. Set to 0 or `u32::MAX` to disable
    /// tiling (request the full image as one region).
    pub tile_size: u32,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
        }
    }
}

impl TileConfig {
    /// Tiling disabled — request the full image as a single region.
    pub fn disabled() -> Self {
        Self { tile_size: 0 }
    }
}

/// Request the full output of a node, optionally using tiled execution.
///
/// When `tile_size` is smaller than the image dimensions, the image is
/// requested as a grid of tiles and stitched into a contiguous buffer.
/// This bounds peak memory to roughly `tile_size² × bpp × pipeline_depth`
/// instead of `width × height × bpp × pipeline_depth`.
///
/// The output is byte-identical to a single full-image request.
fn request_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    width: u32,
    height: u32,
    bpp: u32,
    tile_size: u32,
) -> Result<Vec<u8>, ImageError> {
    // Fall back to full-image request if tiling is disabled or unnecessary
    if tile_size == 0 || (width <= tile_size && height <= tile_size) {
        let full = Rect::new(0, 0, width, height);
        return graph.request_region(node_id, full);
    }

    let stride = width as usize * bpp as usize;
    let mut out = vec![0u8; height as usize * stride];

    let mut y = 0u32;
    while y < height {
        let th = tile_size.min(height - y);
        let mut x = 0u32;
        while x < width {
            let tw = tile_size.min(width - x);
            let tile_rect = Rect::new(x, y, tw, th);
            let tile_pixels = graph.request_region(node_id, tile_rect)?;

            // Stitch tile into output buffer
            let tile_stride = tw as usize * bpp as usize;
            for row in 0..th as usize {
                let dst = (y as usize + row) * stride + x as usize * bpp as usize;
                let src = row * tile_stride;
                out[dst..dst + tile_stride]
                    .copy_from_slice(&tile_pixels[src..src + tile_stride]);
            }

            x += tile_size;
        }
        y += tile_size;
    }

    Ok(out)
}

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
    write_tiled(graph, node_id, format, quality, metadata, &TileConfig::default())
}

/// Write with explicit tile configuration.
pub fn write_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    format: &str,
    quality: Option<u8>,
    metadata: Option<&MetadataSet>,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
    // Re-query info after compute — mapper nodes update their output format
    // during compute_region (e.g., RGB8 → Gray8 for grayscale/sobel/charcoal).
    let info = graph.node_info(node_id)?;
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
    write_jpeg_tiled(graph, node_id, config, metadata, &TileConfig::default())
}

/// Write JPEG with explicit tile configuration.
pub fn write_jpeg_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::jpeg::JpegEncodeConfig,
    metadata: Option<&MetadataSet>,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
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
    write_avif_tiled(graph, node_id, config, metadata, &TileConfig::default())
}

/// Write AVIF with explicit tile configuration.
pub fn write_avif_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::avif::AvifEncodeConfig,
    metadata: Option<&MetadataSet>,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
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
    write_png_tiled(graph, node_id, config, metadata, &TileConfig::default())
}

/// Write PNG with explicit tile configuration.
pub fn write_png_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::png::PngEncodeConfig,
    metadata: Option<&MetadataSet>,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
    let encoded = encoder::png::encode(&pixels, &info, config)?;

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
    write_gif_tiled(graph, node_id, config, metadata, &TileConfig::default())
}

/// Write GIF with explicit tile configuration.
pub fn write_gif_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::gif::GifEncodeConfig,
    metadata: Option<&MetadataSet>,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
    let encoded = encoder::gif::encode_pixels(&pixels, &info, config)?;

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
    write_tiff_tiled(graph, node_id, config, &TileConfig::default())
}

/// Write TIFF with explicit tile configuration.
pub fn write_tiff_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::tiff::TiffEncodeConfig,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
    encoder::tiff::encode(&pixels, &info, config)
}

/// Write a node's output as BMP.
pub fn write_bmp(graph: &mut NodeGraph, node_id: u32) -> Result<Vec<u8>, ImageError> {
    write_bmp_tiled(graph, node_id, &TileConfig::default())
}

/// Write BMP with explicit tile configuration.
pub fn write_bmp_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
    encoder::bmp::encode_pixels(&pixels, &info, &encoder::bmp::BmpEncodeConfig)
}

/// Write a node's output as ICO.
pub fn write_ico(graph: &mut NodeGraph, node_id: u32) -> Result<Vec<u8>, ImageError> {
    write_ico_tiled(graph, node_id, &TileConfig::default())
}

/// Write ICO with explicit tile configuration.
pub fn write_ico_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
    encoder::ico::encode_pixels(&pixels, &info, &encoder::ico::IcoEncodeConfig)
}

/// Write a node's output as QOI.
pub fn write_qoi(graph: &mut NodeGraph, node_id: u32) -> Result<Vec<u8>, ImageError> {
    write_qoi_tiled(graph, node_id, &TileConfig::default())
}

/// Write QOI with explicit tile configuration.
pub fn write_qoi_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
    encoder::encode(&pixels, &info, "qoi", None)
}

/// Write a node's output as WebP with typed config and optional metadata.
pub fn write_webp(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::webp::WebpEncodeConfig,
    metadata: Option<&MetadataSet>,
) -> Result<Vec<u8>, ImageError> {
    write_webp_tiled(graph, node_id, config, metadata, &TileConfig::default())
}

/// Write WebP with explicit tile configuration.
pub fn write_webp_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::webp::WebpEncodeConfig,
    metadata: Option<&MetadataSet>,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let pixels = request_tiled(graph, node_id, info.width, info.height, bpp, tile_config.tile_size)?;
    let encoded = encoder::webp::encode_pixels(&pixels, &info, config)?;

    match metadata {
        Some(ms) if !ms.is_empty() => embed_metadata(encoded, "webp", ms),
        _ => Ok(encoded),
    }
}
