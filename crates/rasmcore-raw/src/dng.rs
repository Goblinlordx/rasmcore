//! DNG (Digital Negative) format decoder.
//!
//! DNG is based on TIFF/EP with additional tags for RAW sensor data.
//! This module reads DNG-specific metadata, extracts raw sensor data,
//! and orchestrates the full decode pipeline:
//! TIFF container → raw extraction → demosaic → color pipeline → RGB output.

use crate::RawError;
use crate::color::{apply_color_pipeline, build_camera_to_srgb};
use crate::demosaic::{CfaPattern, demosaic_bilinear};
use crate::ljpeg;
use crate::tiff::*;

/// Parsed DNG metadata from IFD tags.
#[derive(Debug)]
pub struct DngMetadata {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u16,
    pub compression: u16,
    pub cfa_pattern: CfaPattern,
    /// ColorMatrix1: 3×3, maps XYZ D65 → camera RGB (stored as 9 f64 values, row-major).
    pub color_matrix: [f64; 9],
    /// AsShotNeutral: per-channel neutral values (typically 3 values).
    /// White balance multipliers are derived as 1/neutral (normalized).
    pub as_shot_neutral: [f64; 3],
    pub black_level: f64,
    pub white_level: f64,
    /// Strip offsets and byte counts for raw data extraction.
    pub strip_offsets: Vec<u32>,
    pub strip_byte_counts: Vec<u32>,
    /// Tile dimensions and offsets (if tiled).
    pub tile_width: Option<u32>,
    pub tile_length: Option<u32>,
    pub tile_offsets: Vec<u32>,
    pub tile_byte_counts: Vec<u32>,
    pub rows_per_strip: u32,
    /// Active area [top, left, bottom, right] — crop to this region. (Future: cropping)
    pub _active_area: Option<[u32; 4]>,
    /// Default crop origin. (Future: cropping)
    pub _default_crop_origin: Option<[u32; 2]>,
    /// Default crop size. (Future: cropping)
    pub _default_crop_size: Option<[u32; 2]>,
}

/// Check if data is a DNG file (TIFF header + DNGVersion tag present).
pub fn is_dng(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    // Must have valid TIFF header
    let tiff = match TiffContainer::parse(data) {
        Ok(t) => t,
        Err(_) => return false,
    };

    // Read first IFD and check for DNGVersion tag
    let ifd_offset = tiff.first_ifd_offset();
    let entries = match tiff.read_ifd(ifd_offset) {
        Ok(e) => e,
        Err(_) => return false,
    };

    // Look for DNGVersion tag (50706)
    if let Some(entry) = TiffContainer::find_tag(&entries, TAG_DNG_VERSION) {
        // DNGVersion must be [1, x, x, x]
        if let Ok(data) = tiff.tag_data(entry) {
            return data.len() >= 4 && data[0] == 1;
        }
    }

    false
}

/// Full DNG decode: parse metadata, extract raw data, demosaic, color pipeline.
///
/// Returns (pixels, width, height, pixel_format):
/// - `pixels`: RGB8 byte data (interleaved)
/// - `width`, `height`: output dimensions
/// - `is_16bit`: true if source was 16-bit (output is still 8-bit for compatibility)
pub fn decode_dng(data: &[u8]) -> Result<DngDecodeResult, RawError> {
    let tiff = TiffContainer::parse(data)?;

    // Find the raw image IFD. DNG files typically have:
    // - IFD0: thumbnail or full-size preview
    // - SubIFD: raw sensor data (NewSubFileType = 0)
    // We need the IFD with CFA data.
    let metadata = find_raw_ifd_and_parse(&tiff)?;

    // Extract raw sensor data
    let raw_u16 = extract_raw_data(&tiff, &metadata)?;

    // Demosaic
    let rgb16 = demosaic_bilinear(
        &raw_u16,
        metadata.width,
        metadata.height,
        metadata.cfa_pattern,
    )?;

    // Build color pipeline
    let camera_to_srgb = build_camera_to_srgb(&metadata.color_matrix)
        .ok_or_else(|| RawError::InvalidFormat("singular ColorMatrix1 — cannot invert".into()))?;

    // Compute white balance multipliers from AsShotNeutral
    let wb = compute_white_balance(&metadata.as_shot_neutral);

    // Apply color pipeline → RGB8 output
    let pixels = apply_color_pipeline(
        &rgb16,
        metadata.width,
        metadata.height,
        &camera_to_srgb,
        &wb,
        metadata.black_level,
        metadata.white_level,
        true, // 8-bit output
    );

    Ok(DngDecodeResult {
        pixels,
        width: metadata.width,
        height: metadata.height,
        bits_per_sample: metadata.bits_per_sample,
    })
}

/// Full DNG decode with 16-bit output.
pub fn decode_dng_16bit(data: &[u8]) -> Result<DngDecodeResult, RawError> {
    let tiff = TiffContainer::parse(data)?;
    let metadata = find_raw_ifd_and_parse(&tiff)?;
    let raw_u16 = extract_raw_data(&tiff, &metadata)?;
    let rgb16 = demosaic_bilinear(
        &raw_u16,
        metadata.width,
        metadata.height,
        metadata.cfa_pattern,
    )?;

    let camera_to_srgb = build_camera_to_srgb(&metadata.color_matrix)
        .ok_or_else(|| RawError::InvalidFormat("singular ColorMatrix1 — cannot invert".into()))?;
    let wb = compute_white_balance(&metadata.as_shot_neutral);

    let pixels = apply_color_pipeline(
        &rgb16,
        metadata.width,
        metadata.height,
        &camera_to_srgb,
        &wb,
        metadata.black_level,
        metadata.white_level,
        false, // 16-bit output
    );

    Ok(DngDecodeResult {
        pixels,
        width: metadata.width,
        height: metadata.height,
        bits_per_sample: metadata.bits_per_sample,
    })
}

/// Result of DNG decoding.
pub struct DngDecodeResult {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u16,
}

/// Compute white balance multipliers from AsShotNeutral.
/// AsShotNeutral gives the per-channel response to a neutral color.
/// WB multipliers = 1/neutral, normalized so the green channel = 1.0.
fn compute_white_balance(neutral: &[f64; 3]) -> [f64; 3] {
    if neutral[0] == 0.0 || neutral[1] == 0.0 || neutral[2] == 0.0 {
        return [1.0, 1.0, 1.0]; // fallback: no WB adjustment
    }
    let inv = [1.0 / neutral[0], 1.0 / neutral[1], 1.0 / neutral[2]];
    // Normalize so green (index 1) = 1.0
    let g = inv[1];
    [inv[0] / g, 1.0, inv[2] / g]
}

/// Find the raw image IFD and parse DNG metadata.
fn find_raw_ifd_and_parse(tiff: &TiffContainer<'_>) -> Result<DngMetadata, RawError> {
    let ifd0_offset = tiff.first_ifd_offset();
    let entries = tiff.read_ifd(ifd0_offset)?;

    // Check if IFD0 has raw data (NewSubFileType = 0 or absent, and has CFA)
    if let Some(meta) = try_parse_raw_ifd(tiff, &entries)? {
        return Ok(meta);
    }

    // Check SubIFDs
    if let Some(sub_entry) = TiffContainer::find_tag(&entries, TAG_SUB_IFDS) {
        let sub_offsets = tiff.tag_u32_vec(sub_entry)?;
        for offset in sub_offsets {
            let sub_entries = tiff.read_ifd(offset)?;
            if let Some(meta) = try_parse_raw_ifd(tiff, &sub_entries)? {
                return Ok(meta);
            }
        }
    }

    // Try next IFD in chain
    let next_offset = tiff.next_ifd_offset(ifd0_offset)?;
    if next_offset != 0 {
        let next_entries = tiff.read_ifd(next_offset)?;
        if let Some(meta) = try_parse_raw_ifd(tiff, &next_entries)? {
            return Ok(meta);
        }
    }

    // If we haven't found CFA data, try IFD0 as a last resort (some DNGs put
    // everything in IFD0 without SubIFDs)
    parse_dng_metadata(tiff, &entries)
}

/// Try to parse an IFD as raw CFA data. Returns None if it's not a raw IFD.
fn try_parse_raw_ifd(
    tiff: &TiffContainer<'_>,
    entries: &[IfdEntry],
) -> Result<Option<DngMetadata>, RawError> {
    // Check NewSubFileType — raw data has type 0 (full-resolution image)
    if let Some(nsft) = TiffContainer::find_tag(entries, TAG_NEW_SUBFILE_TYPE) {
        let val = tiff.tag_u32(nsft)?;
        if val != 0 {
            return Ok(None); // This is a thumbnail or reduced-resolution image
        }
    }

    // Check for CFA-related tags that indicate this is raw sensor data
    let has_cfa = TiffContainer::find_tag(entries, TAG_CFA_PATTERN).is_some();
    if !has_cfa {
        return Ok(None);
    }

    parse_dng_metadata(tiff, entries).map(Some)
}

/// Parse DNG metadata from an IFD's entries.
fn parse_dng_metadata(
    tiff: &TiffContainer<'_>,
    entries: &[IfdEntry],
) -> Result<DngMetadata, RawError> {
    // Required: width, height
    let width = TiffContainer::find_tag(entries, TAG_IMAGE_WIDTH)
        .ok_or_else(|| RawError::InvalidFormat("missing ImageWidth tag".into()))
        .and_then(|e| tiff.tag_u32(e))?;
    let height = TiffContainer::find_tag(entries, TAG_IMAGE_LENGTH)
        .ok_or_else(|| RawError::InvalidFormat("missing ImageLength tag".into()))
        .and_then(|e| tiff.tag_u32(e))?;

    // BitsPerSample (default 8)
    let bits_per_sample = TiffContainer::find_tag(entries, TAG_BITS_PER_SAMPLE)
        .map(|e| tiff.tag_u16(e))
        .transpose()?
        .unwrap_or(8);

    // Compression (1 = none, 7 = lossless JPEG)
    let compression = TiffContainer::find_tag(entries, TAG_COMPRESSION)
        .map(|e| tiff.tag_u16(e))
        .transpose()?
        .unwrap_or(COMPRESSION_NONE);

    // CFA Pattern
    let cfa_entry = TiffContainer::find_tag(entries, TAG_CFA_PATTERN)
        .ok_or_else(|| RawError::InvalidFormat("missing CFAPattern tag".into()))?;
    let cfa_data = tiff.tag_data(cfa_entry)?;
    let cfa_pattern = CfaPattern::from_cfa_bytes(cfa_data)?;

    // ColorMatrix1 (required for color pipeline)
    let color_matrix = if let Some(cm_entry) = TiffContainer::find_tag(entries, TAG_COLOR_MATRIX_1)
    {
        let vals = tiff.tag_srational_vec(cm_entry)?;
        if vals.len() < 9 {
            return Err(RawError::InvalidFormat(format!(
                "ColorMatrix1 needs 9 values, got {}",
                vals.len()
            )));
        }
        [
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8],
        ]
    } else {
        // Fallback: identity (no color correction)
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    };

    // AsShotNeutral
    let as_shot_neutral =
        if let Some(asn_entry) = TiffContainer::find_tag(entries, TAG_AS_SHOT_NEUTRAL) {
            let vals = tiff.tag_rational_vec(asn_entry)?;
            if vals.len() >= 3 {
                [vals[0], vals[1], vals[2]]
            } else {
                [1.0, 1.0, 1.0]
            }
        } else {
            [1.0, 1.0, 1.0]
        };

    // BlackLevel (default 0)
    let black_level = TiffContainer::find_tag(entries, TAG_BLACK_LEVEL)
        .map(|e| -> Result<f64, RawError> {
            match e.tag_type {
                TagType::Short => Ok(tiff.tag_u16(e)? as f64),
                TagType::Long => Ok(tiff.tag_u32(e)? as f64),
                TagType::Rational => {
                    let vals = tiff.tag_rational_vec(e)?;
                    Ok(vals.first().copied().unwrap_or(0.0))
                }
                _ => Ok(0.0),
            }
        })
        .transpose()?
        .unwrap_or(0.0);

    // WhiteLevel (default = 2^bits - 1)
    let white_level = TiffContainer::find_tag(entries, TAG_WHITE_LEVEL)
        .map(|e| -> Result<f64, RawError> {
            match e.tag_type {
                TagType::Short => Ok(tiff.tag_u16(e)? as f64),
                TagType::Long => Ok(tiff.tag_u32(e)? as f64),
                _ => Ok(((1u32 << bits_per_sample) - 1) as f64),
            }
        })
        .transpose()?
        .unwrap_or(((1u32 << bits_per_sample) - 1) as f64);

    // Strip offsets/byte counts
    let strip_offsets = TiffContainer::find_tag(entries, TAG_STRIP_OFFSETS)
        .map(|e| tiff.tag_u32_vec(e))
        .transpose()?
        .unwrap_or_default();
    let strip_byte_counts = TiffContainer::find_tag(entries, TAG_STRIP_BYTE_COUNTS)
        .map(|e| tiff.tag_u32_vec(e))
        .transpose()?
        .unwrap_or_default();

    // Tile info
    let tile_width = TiffContainer::find_tag(entries, TAG_TILE_WIDTH)
        .map(|e| tiff.tag_u32(e))
        .transpose()?;
    let tile_length = TiffContainer::find_tag(entries, TAG_TILE_LENGTH)
        .map(|e| tiff.tag_u32(e))
        .transpose()?;
    let tile_offsets = TiffContainer::find_tag(entries, TAG_TILE_OFFSETS)
        .map(|e| tiff.tag_u32_vec(e))
        .transpose()?
        .unwrap_or_default();
    let tile_byte_counts = TiffContainer::find_tag(entries, TAG_TILE_BYTE_COUNTS)
        .map(|e| tiff.tag_u32_vec(e))
        .transpose()?
        .unwrap_or_default();

    // RowsPerStrip (default = height)
    let rows_per_strip = TiffContainer::find_tag(entries, TAG_ROWS_PER_STRIP)
        .map(|e| tiff.tag_u32(e))
        .transpose()?
        .unwrap_or(height);

    // Active area
    let active_area = TiffContainer::find_tag(entries, TAG_ACTIVE_AREA)
        .map(|e| -> Result<[u32; 4], RawError> {
            let vals = tiff.tag_u32_vec(e)?;
            if vals.len() >= 4 {
                Ok([vals[0], vals[1], vals[2], vals[3]])
            } else {
                Ok([0, 0, height, width])
            }
        })
        .transpose()?;

    // Default crop
    let default_crop_origin = TiffContainer::find_tag(entries, TAG_DEFAULT_CROP_ORIGIN)
        .map(|e| -> Result<[u32; 2], RawError> {
            match e.tag_type {
                TagType::Rational => {
                    let vals = tiff.tag_rational_vec(e)?;
                    if vals.len() >= 2 {
                        Ok([vals[0] as u32, vals[1] as u32])
                    } else {
                        Ok([0, 0])
                    }
                }
                _ => {
                    let vals = tiff.tag_u32_vec(e)?;
                    if vals.len() >= 2 {
                        Ok([vals[0], vals[1]])
                    } else {
                        Ok([0, 0])
                    }
                }
            }
        })
        .transpose()?;

    let default_crop_size = TiffContainer::find_tag(entries, TAG_DEFAULT_CROP_SIZE)
        .map(|e| -> Result<[u32; 2], RawError> {
            match e.tag_type {
                TagType::Rational => {
                    let vals = tiff.tag_rational_vec(e)?;
                    if vals.len() >= 2 {
                        Ok([vals[0] as u32, vals[1] as u32])
                    } else {
                        Ok([width, height])
                    }
                }
                _ => {
                    let vals = tiff.tag_u32_vec(e)?;
                    if vals.len() >= 2 {
                        Ok([vals[0], vals[1]])
                    } else {
                        Ok([width, height])
                    }
                }
            }
        })
        .transpose()?;

    Ok(DngMetadata {
        width,
        height,
        bits_per_sample,
        compression,
        cfa_pattern,
        color_matrix,
        as_shot_neutral,
        black_level,
        white_level,
        strip_offsets,
        strip_byte_counts,
        tile_width,
        tile_length,
        tile_offsets,
        tile_byte_counts,
        rows_per_strip,
        _active_area: active_area,
        _default_crop_origin: default_crop_origin,
        _default_crop_size: default_crop_size,
    })
}

/// Extract raw sensor data as a flat u16 array (one value per pixel).
fn extract_raw_data(tiff: &TiffContainer<'_>, meta: &DngMetadata) -> Result<Vec<u16>, RawError> {
    let w = meta.width as usize;
    let h = meta.height as usize;

    match meta.compression {
        COMPRESSION_NONE => extract_uncompressed(tiff, meta, w, h),
        COMPRESSION_LOSSLESS_JPEG => extract_lossless_jpeg(tiff, meta, w, h),
        other => Err(RawError::UnsupportedCompression(other)),
    }
}

/// Extract uncompressed raw data from strips or tiles.
fn extract_uncompressed(
    tiff: &TiffContainer<'_>,
    meta: &DngMetadata,
    w: usize,
    h: usize,
) -> Result<Vec<u16>, RawError> {
    let mut raw = vec![0u16; w * h];
    let bps = meta.bits_per_sample;

    if !meta.tile_offsets.is_empty() {
        // Tiled layout
        extract_tiles_uncompressed(tiff, meta, &mut raw, w, h, bps)?;
    } else if !meta.strip_offsets.is_empty() {
        // Strip layout
        extract_strips_uncompressed(tiff, meta, &mut raw, w, h, bps)?;
    } else {
        return Err(RawError::InvalidFormat(
            "no strip or tile offsets found".into(),
        ));
    }

    Ok(raw)
}

fn extract_strips_uncompressed(
    tiff: &TiffContainer<'_>,
    meta: &DngMetadata,
    raw: &mut [u16],
    w: usize,
    _h: usize,
    bps: u16,
) -> Result<(), RawError> {
    let rps = meta.rows_per_strip as usize;
    let mut pixel_idx = 0usize;

    for (i, (&offset, &byte_count)) in meta
        .strip_offsets
        .iter()
        .zip(meta.strip_byte_counts.iter())
        .enumerate()
    {
        let strip_data = tiff.raw_data(offset, byte_count)?;
        let rows_in_strip = rps.min(raw.len() / w - i * rps);

        read_pixels_from_bytes(
            strip_data,
            &mut raw[pixel_idx..],
            w * rows_in_strip,
            bps,
            tiff.order,
        )?;
        pixel_idx += w * rows_in_strip;
    }

    Ok(())
}

fn extract_tiles_uncompressed(
    tiff: &TiffContainer<'_>,
    meta: &DngMetadata,
    raw: &mut [u16],
    w: usize,
    h: usize,
    bps: u16,
) -> Result<(), RawError> {
    let tw = meta.tile_width.unwrap_or(w as u32) as usize;
    let tl = meta.tile_length.unwrap_or(h as u32) as usize;
    let tiles_across = w.div_ceil(tw);

    for (i, (&offset, &byte_count)) in meta
        .tile_offsets
        .iter()
        .zip(meta.tile_byte_counts.iter())
        .enumerate()
    {
        let tile_data = tiff.raw_data(offset, byte_count)?;
        let tile_col = (i % tiles_across) * tw;
        let tile_row = (i / tiles_across) * tl;

        let mut tile_pixels = vec![0u16; tw * tl];
        read_pixels_from_bytes(tile_data, &mut tile_pixels, tw * tl, bps, tiff.order)?;

        // Copy tile into output
        for row in 0..tl {
            let out_row = tile_row + row;
            if out_row >= h {
                break;
            }
            let src_start = row * tw;
            let dst_start = out_row * w + tile_col;
            let copy_w = tw.min(w - tile_col);
            raw[dst_start..dst_start + copy_w]
                .copy_from_slice(&tile_pixels[src_start..src_start + copy_w]);
        }
    }

    Ok(())
}

/// Read pixel values from raw bytes into u16 array, handling different bit depths.
fn read_pixels_from_bytes(
    data: &[u8],
    output: &mut [u16],
    count: usize,
    bps: u16,
    order: ByteOrder,
) -> Result<(), RawError> {
    let count = count.min(output.len());

    match bps {
        8 => {
            for (out, &byte) in output.iter_mut().zip(data.iter()).take(count) {
                *out = byte as u16;
            }
        }
        16 => {
            for (i, out) in output.iter_mut().enumerate().take(count) {
                let offset = i * 2;
                if offset + 1 >= data.len() {
                    break;
                }
                *out = order.u16(&data[offset..offset + 2]);
            }
        }
        12 => {
            // 12-bit packed: 2 pixels in 3 bytes
            let mut out_idx = 0;
            let mut byte_idx = 0;
            while out_idx + 1 < count && byte_idx + 2 < data.len() {
                let b0 = data[byte_idx] as u16;
                let b1 = data[byte_idx + 1] as u16;
                let b2 = data[byte_idx + 2] as u16;
                // First pixel: upper 8 bits of b0, upper 4 bits of b1
                output[out_idx] = (b0 << 4) | (b1 >> 4);
                // Second pixel: lower 4 bits of b1, all 8 bits of b2
                if out_idx + 1 < count {
                    output[out_idx + 1] = ((b1 & 0x0F) << 8) | b2;
                }
                out_idx += 2;
                byte_idx += 3;
            }
        }
        14 => {
            // 14-bit packed: 4 pixels in 7 bytes
            let mut out_idx = 0;
            let mut byte_idx = 0;
            while out_idx + 3 < count && byte_idx + 6 < data.len() {
                let mut bits = 0u64;
                for k in 0..7 {
                    bits = (bits << 8) | data[byte_idx + k] as u64;
                }
                // 56 bits = 4 × 14 bits
                output[out_idx] = ((bits >> 42) & 0x3FFF) as u16;
                output[out_idx + 1] = ((bits >> 28) & 0x3FFF) as u16;
                output[out_idx + 2] = ((bits >> 14) & 0x3FFF) as u16;
                output[out_idx + 3] = (bits & 0x3FFF) as u16;
                out_idx += 4;
                byte_idx += 7;
            }
        }
        _ => {
            return Err(RawError::InvalidFormat(format!(
                "unsupported BitsPerSample: {bps}"
            )));
        }
    }

    Ok(())
}

/// Extract lossless JPEG compressed raw data.
fn extract_lossless_jpeg(
    tiff: &TiffContainer<'_>,
    meta: &DngMetadata,
    w: usize,
    h: usize,
) -> Result<Vec<u16>, RawError> {
    let mut raw = vec![0u16; w * h];

    if !meta.tile_offsets.is_empty() {
        // Tiled layout — each tile is a separate lossless JPEG
        let tw = meta.tile_width.unwrap_or(w as u32) as usize;
        let tl = meta.tile_length.unwrap_or(h as u32) as usize;
        let tiles_across = w.div_ceil(tw);

        for (i, (&offset, &byte_count)) in meta
            .tile_offsets
            .iter()
            .zip(meta.tile_byte_counts.iter())
            .enumerate()
        {
            let tile_data = tiff.raw_data(offset, byte_count)?;
            let ljpeg_result = ljpeg::decode_ljpeg(tile_data)?;

            let tile_col = (i % tiles_across) * tw;
            let tile_row = (i / tiles_across) * tl;

            // The lossless JPEG may have 2 components (for DNG interleaved CFA).
            // If 2 components: width in LJPEG = actual_tile_width / 2, and pixels
            // are interleaved [c0, c1] representing adjacent Bayer pairs.
            let ljw = ljpeg_result.width as usize;
            let ljh = ljpeg_result.height as usize;
            let ljc = ljpeg_result.components as usize;

            for row in 0..ljh {
                let out_row = tile_row + row;
                if out_row >= h {
                    break;
                }
                for col in 0..ljw {
                    for c in 0..ljc {
                        let src_idx = (row * ljw + col) * ljc + c;
                        let out_col = tile_col + col * ljc + c;
                        if out_col < w && src_idx < ljpeg_result.data.len() {
                            raw[out_row * w + out_col] = ljpeg_result.data[src_idx];
                        }
                    }
                }
            }
        }
    } else if !meta.strip_offsets.is_empty() {
        // Strip layout — single or multiple lossless JPEG segments
        let rps = meta.rows_per_strip as usize;

        for (i, (&offset, &byte_count)) in meta
            .strip_offsets
            .iter()
            .zip(meta.strip_byte_counts.iter())
            .enumerate()
        {
            let strip_data = tiff.raw_data(offset, byte_count)?;
            let ljpeg_result = ljpeg::decode_ljpeg(strip_data)?;

            let start_row = i * rps;
            let ljw = ljpeg_result.width as usize;
            let ljh = ljpeg_result.height as usize;
            let ljc = ljpeg_result.components as usize;

            for row in 0..ljh {
                let out_row = start_row + row;
                if out_row >= h {
                    break;
                }
                for col in 0..ljw {
                    for c in 0..ljc {
                        let src_idx = (row * ljw + col) * ljc + c;
                        let out_col = col * ljc + c;
                        if out_col < w && src_idx < ljpeg_result.data.len() {
                            raw[out_row * w + out_col] = ljpeg_result.data[src_idx];
                        }
                    }
                }
            }
        }
    } else {
        return Err(RawError::InvalidFormat(
            "no strip or tile offsets for lossless JPEG data".into(),
        ));
    }

    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_dng_false_for_regular_tiff() {
        // Regular TIFF (no DNGVersion tag)
        let mut data = Vec::new();
        // TIFF header: little-endian, magic 42, IFD at offset 8
        data.extend_from_slice(b"II");
        data.extend_from_slice(&42u16.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        // IFD: 1 entry
        data.extend_from_slice(&1u16.to_le_bytes());
        // Entry: ImageWidth = 100
        data.extend_from_slice(&256u16.to_le_bytes());
        data.extend_from_slice(&3u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&100u16.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        // Next IFD = 0
        data.extend_from_slice(&0u32.to_le_bytes());

        assert!(!is_dng(&data));
    }

    #[test]
    fn is_dng_true_for_dng_header() {
        let mut data = Vec::new();
        // TIFF header: little-endian, magic 42, IFD at offset 8
        data.extend_from_slice(b"II");
        data.extend_from_slice(&42u16.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        // IFD: 1 entry (DNGVersion)
        data.extend_from_slice(&1u16.to_le_bytes());
        // Entry: DNGVersion (50706) = [1, 4, 0, 0] (type BYTE, count 4)
        data.extend_from_slice(&50706u16.to_le_bytes()); // tag
        data.extend_from_slice(&1u16.to_le_bytes()); // type BYTE
        data.extend_from_slice(&4u32.to_le_bytes()); // count
        data.push(1); // version[0]
        data.push(4); // version[1]
        data.push(0); // version[2]
        data.push(0); // version[3]
        // Next IFD = 0
        data.extend_from_slice(&0u32.to_le_bytes());

        assert!(is_dng(&data));
    }

    #[test]
    fn white_balance_neutral() {
        let wb = compute_white_balance(&[1.0, 1.0, 1.0]);
        assert!((wb[0] - 1.0).abs() < 1e-10);
        assert!((wb[1] - 1.0).abs() < 1e-10);
        assert!((wb[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn white_balance_typical() {
        // Typical DNG AsShotNeutral values
        let wb = compute_white_balance(&[0.4, 1.0, 0.6]);
        // R gain = 1/0.4 / (1/1.0) = 2.5
        // G gain = 1.0
        // B gain = 1/0.6 / (1/1.0) ≈ 1.667
        assert!((wb[0] - 2.5).abs() < 1e-10);
        assert!((wb[1] - 1.0).abs() < 1e-10);
        assert!((wb[2] - 1.0 / 0.6).abs() < 1e-10);
    }

    #[test]
    fn read_pixels_8bit() {
        let data = [100u8, 200, 50, 150];
        let mut output = [0u16; 4];
        read_pixels_from_bytes(&data, &mut output, 4, 8, ByteOrder::Little).unwrap();
        assert_eq!(output, [100, 200, 50, 150]);
    }

    #[test]
    fn read_pixels_16bit_le() {
        let data = [0x00, 0x10, 0xFF, 0x0F]; // 4096, 4095 in LE
        let mut output = [0u16; 2];
        read_pixels_from_bytes(&data, &mut output, 2, 16, ByteOrder::Little).unwrap();
        assert_eq!(output, [4096, 4095]);
    }

    #[test]
    fn read_pixels_12bit() {
        // 12-bit packed: 2 pixels in 3 bytes
        // Pixel 0: 0xABC (2748), Pixel 1: 0xDEF (3567)
        // Byte 0: upper 8 of pixel 0 = 0xAB
        // Byte 1: lower 4 of pixel 0 (C) << 4 | upper 4 of pixel 1 (D) = 0xCD
        // Byte 2: lower 8 of pixel 1 = 0xEF
        let data = [0xAB, 0xCD, 0xEF];
        let mut output = [0u16; 2];
        read_pixels_from_bytes(&data, &mut output, 2, 12, ByteOrder::Little).unwrap();
        assert_eq!(output[0], 0xABC);
        assert_eq!(output[1], 0xDEF);
    }
}
