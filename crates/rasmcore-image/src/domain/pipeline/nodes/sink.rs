//! Sink nodes — drive execution by pulling regions and encoding output.
//!
//! By default, the output image is requested as tiles (default 512×512) to
//! reduce peak memory for large images. For formats with streaming encoders
//! (BMP, HDR, FITS, QOI, TIFF), tiles are passed directly to the encoder
//! without assembling the full pixel buffer. For other formats, tiles are
//! stitched into a contiguous buffer before encoding.
//!
//! Tile execution is pixel-identical to full-image execution — the pipeline
//! graph's `request_region` produces the same bytes regardless of request
//! size. The spatial cache automatically reuses overlapping tile edges.
//!
//! Streaming encoder memory benefit:
//!   - BMP, HDR, FITS: true streaming — no full pixel buffer in memory
//!   - QOI, TIFF: buffered internally — API consistency, no memory win
//!   - PNG, JPEG: scanline-based streaming possible (future work)
//!   - WebP, AVIF, GIF: require complete pixel buffers (no streaming)

use crate::domain::encoder;
use crate::domain::encoder::streaming::StreamingEncoder;
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

/// Feed tiles from the pipeline graph to a streaming encoder.
///
/// Like `request_tiled` but passes tiles directly to the encoder instead
/// of stitching into an intermediate buffer. The encoder handles format-
/// specific assembly.
fn feed_tiles_to_encoder(
    graph: &mut NodeGraph,
    node_id: u32,
    width: u32,
    height: u32,
    _bpp: u32,
    tile_size: u32,
    enc: &mut dyn StreamingEncoder,
) -> Result<(), ImageError> {
    if tile_size == 0 || (width <= tile_size && height <= tile_size) {
        let full = Rect::new(0, 0, width, height);
        let pixels = graph.request_region(node_id, full)?;
        enc.write_tile(&pixels, 0, 0, width, height)?;
        return Ok(());
    }

    let mut y = 0u32;
    while y < height {
        let th = tile_size.min(height - y);
        let mut x = 0u32;
        while x < width {
            let tw = tile_size.min(width - x);
            let tile_rect = Rect::new(x, y, tw, th);
            let tile_pixels = graph.request_region(node_id, tile_rect)?;
            enc.write_tile(&tile_pixels, x, y, tw, th)?;
            x += tile_size;
        }
        y += tile_size;
    }
    Ok(())
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
///
/// Uses a streaming encoder when available for the format, avoiding the
/// intermediate full pixel buffer. Falls back to stitch-then-encode for
/// formats without streaming support.
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

    // Try streaming path for supported formats (no metadata embedding needed
    // for BMP/HDR/FITS/QOI/TIFF — they don't support metadata embedding)
    let no_metadata = metadata.is_none()
        || matches!(
            format,
            "bmp" | "hdr" | "fits" | "fit" | "qoi" | "tiff" | "tif"
        );
    if no_metadata
        && let Some(mut enc) = encoder::streaming::create_streaming_encoder(format, &info)
    {
        feed_tiles_to_encoder(
            graph,
            node_id,
            info.width,
            info.height,
            bpp,
            tile_config.tile_size,
            enc.as_mut(),
        )?;
        return enc.finish();
    }

    // Fallback: stitch tiles into full buffer, then encode
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

/// Write TIFF with explicit tile configuration (streaming — internally buffered).
pub fn write_tiff_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    config: &encoder::tiff::TiffEncodeConfig,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let mut enc = encoder::streaming::TiffStreamingEncoder::new(&info, config)?;
    feed_tiles_to_encoder(graph, node_id, info.width, info.height, bpp, tile_config.tile_size, &mut enc)?;
    enc.finish()
}

/// Write a node's output as BMP.
pub fn write_bmp(graph: &mut NodeGraph, node_id: u32) -> Result<Vec<u8>, ImageError> {
    write_bmp_tiled(graph, node_id, &TileConfig::default())
}

/// Write BMP with explicit tile configuration (streaming).
pub fn write_bmp_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let mut enc = encoder::streaming::BmpStreamingEncoder::new(&info)?;
    feed_tiles_to_encoder(graph, node_id, info.width, info.height, bpp, tile_config.tile_size, &mut enc)?;
    enc.finish()
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

/// Write QOI with explicit tile configuration (streaming — internally buffered).
#[cfg(feature = "native-qoi")]
pub fn write_qoi_tiled(
    graph: &mut NodeGraph,
    node_id: u32,
    tile_config: &TileConfig,
) -> Result<Vec<u8>, ImageError> {
    let info = graph.node_info(node_id)?;
    let bpp = bytes_per_pixel(info.format);
    let mut enc = encoder::streaming::QoiStreamingEncoder::new(&info)?;
    feed_tiles_to_encoder(graph, node_id, info.width, info.height, bpp, tile_config.tile_size, &mut enc)?;
    enc.finish()
}

/// Write QOI with explicit tile configuration (fallback when native-qoi disabled).
#[cfg(not(feature = "native-qoi"))]
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

#[cfg(test)]
mod tiled_sink_tests {
    use super::*;
    use crate::domain::pipeline::graph::{crop_region, ImageNode, NodeGraph};
    use crate::domain::types::*;
    use rasmcore_pipeline::rect::Rect;

    /// Raw pixel source that serves sub-regions on demand.
    struct RawSource {
        pixels: Vec<u8>,
        info: ImageInfo,
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
        fn access_pattern(&self) -> crate::domain::pipeline::graph::AccessPattern {
            crate::domain::pipeline::graph::AccessPattern::Sequential
        }
    }

    fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push(((x * 255) / w.max(1)) as u8);
                px.push(((y * 255) / h.max(1)) as u8);
                px.push(128);
            }
        }
        px
    }

    fn make_graph(w: u32, h: u32) -> (NodeGraph, u32) {
        let pixels = gradient_rgb(w, h);
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let mut g = NodeGraph::new(16 * 1024 * 1024);
        let src = g.add_node(Box::new(RawSource { pixels, info }));
        (g, src)
    }

    #[test]
    fn request_tiled_identity_with_full() {
        let w = 100u32;
        let h = 80u32;
        let bpp = 3u32;

        // Full-image request
        let (mut g1, src1) = make_graph(w, h);
        let full = g1.request_region(src1, Rect::new(0, 0, w, h)).unwrap();

        // Tiled request with tile_size=32 (non-evenly-divisible)
        let (mut g2, src2) = make_graph(w, h);
        let tiled = request_tiled(&mut g2, src2, w, h, bpp, 32).unwrap();

        assert_eq!(full.len(), tiled.len());
        assert_eq!(full, tiled, "tiled output must be byte-identical to full");
    }

    #[test]
    fn request_tiled_various_sizes() {
        let w = 100u32;
        let h = 100u32;
        let bpp = 3u32;

        let (mut g_ref, src_ref) = make_graph(w, h);
        let reference = g_ref.request_region(src_ref, Rect::new(0, 0, w, h)).unwrap();

        for tile_size in [1, 7, 32, 50, 64, 99, 100, 200, 512] {
            let (mut g, src) = make_graph(w, h);
            let result = request_tiled(&mut g, src, w, h, bpp, tile_size).unwrap();
            assert_eq!(
                reference, result,
                "tile_size={tile_size} produced different output"
            );
        }
    }

    #[test]
    fn request_tiled_disabled() {
        let w = 50u32;
        let h = 50u32;
        let bpp = 3u32;

        // tile_size=0 should fall back to full-image request
        let (mut g1, src1) = make_graph(w, h);
        let full = g1.request_region(src1, Rect::new(0, 0, w, h)).unwrap();

        let (mut g2, src2) = make_graph(w, h);
        let result = request_tiled(&mut g2, src2, w, h, bpp, 0).unwrap();
        assert_eq!(full, result);
    }

    #[test]
    fn tile_config_default_is_512() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.tile_size, 512);
    }

    #[test]
    fn tile_config_disabled_is_zero() {
        let cfg = TileConfig::disabled();
        assert_eq!(cfg.tile_size, 0);
    }
}
