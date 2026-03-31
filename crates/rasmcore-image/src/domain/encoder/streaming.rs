//! Streaming encoders — accept pixel tiles incrementally to avoid buffering
//! the full decoded pixel buffer.
//!
//! Tiles must arrive in raster order (top-to-bottom, left-to-right within
//! each tile row). The trait is used by the sink's tiled execution path.
//!
//! Streaming benefit varies by format:
//!   - BMP, HDR, FITS: true streaming — no full pixel buffer needed
//!   - QOI, TIFF: buffered internally — API consistency, no memory benefit

use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

/// Trait for encoders that accept pixel tiles incrementally.
pub trait StreamingEncoder {
    /// Accept a tile at position (x, y) with dimensions (w, h).
    fn write_tile(
        &mut self,
        pixels: &[u8],
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), ImageError>;

    /// Finalize encoding and return the complete encoded output.
    fn finish(&mut self) -> Result<Vec<u8>, ImageError>;
}

// ─── BMP Streaming Encoder ─────────────────────────────────────────────────
//
// True streaming: pre-allocates the output buffer (header + pixel area) and
// writes BGR rows directly at the correct offsets. No intermediate pixel
// buffer. BMP stores rows bottom-to-top, BGR order.

pub struct BmpStreamingEncoder {
    buf: Vec<u8>,
    #[allow(dead_code)]
    width: u32,
    height: u32,
    channels: usize,
    row_stride: usize,
    data_offset: usize,
}

impl BmpStreamingEncoder {
    pub fn new(info: &ImageInfo) -> Result<Self, ImageError> {
        let channels = match info.format {
            PixelFormat::Rgb8 => 3usize,
            PixelFormat::Gray8 => return Self::new_gray(info),
            other => {
                return Err(ImageError::UnsupportedFormat(format!(
                    "BMP streaming encode does not support {other:?}"
                )));
            }
        };

        let w = info.width as usize;
        let h = info.height as usize;
        let row_stride = (w * channels + 3) & !3;
        let pixel_data_size = row_stride * h;
        let data_offset = 14 + 40; // file header + info header
        let file_size = data_offset + pixel_data_size;

        let mut buf = vec![0u8; file_size];

        // File header (14 bytes)
        buf[0..2].copy_from_slice(b"BM");
        buf[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
        // reserved = 0 (already zeroed)
        buf[10..14].copy_from_slice(&(data_offset as u32).to_le_bytes());

        // BITMAPINFOHEADER (40 bytes)
        buf[14..18].copy_from_slice(&40u32.to_le_bytes());
        buf[18..22].copy_from_slice(&(info.width as i32).to_le_bytes());
        buf[22..26].copy_from_slice(&(info.height as i32).to_le_bytes());
        buf[26..28].copy_from_slice(&1u16.to_le_bytes()); // planes
        buf[28..30].copy_from_slice(&((channels * 8) as u16).to_le_bytes());
        // compression = BI_RGB = 0 (already zeroed)
        buf[34..38].copy_from_slice(&(pixel_data_size as u32).to_le_bytes());
        buf[38..42].copy_from_slice(&2835u32.to_le_bytes()); // h_res
        buf[42..46].copy_from_slice(&2835u32.to_le_bytes()); // v_res
        // colors_used = 0, colors_important = 0 (already zeroed)

        Ok(Self {
            buf,
            width: info.width,
            height: info.height,
            channels,
            row_stride,
            data_offset,
        })
    }

    fn new_gray(info: &ImageInfo) -> Result<Self, ImageError> {
        let w = info.width as usize;
        let h = info.height as usize;
        let row_stride = (w + 3) & !3;
        let pixel_data_size = row_stride * h;
        let palette_size = 256 * 4;
        let data_offset = 14 + 40 + palette_size;
        let file_size = data_offset + pixel_data_size;

        let mut buf = vec![0u8; file_size];

        // File header
        buf[0..2].copy_from_slice(b"BM");
        buf[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
        buf[10..14].copy_from_slice(&(data_offset as u32).to_le_bytes());

        // Info header
        buf[14..18].copy_from_slice(&40u32.to_le_bytes());
        buf[18..22].copy_from_slice(&(info.width as i32).to_le_bytes());
        buf[22..26].copy_from_slice(&(info.height as i32).to_le_bytes());
        buf[26..28].copy_from_slice(&1u16.to_le_bytes());
        buf[28..30].copy_from_slice(&8u16.to_le_bytes()); // 8 bpp
        buf[34..38].copy_from_slice(&(pixel_data_size as u32).to_le_bytes());
        buf[38..42].copy_from_slice(&2835u32.to_le_bytes());
        buf[42..46].copy_from_slice(&2835u32.to_le_bytes());
        buf[46..50].copy_from_slice(&256u32.to_le_bytes()); // colors_used

        // Grayscale palette: (i, i, i, 0) for i in 0..256
        let palette_start = 54;
        for i in 0..256usize {
            let off = palette_start + i * 4;
            let v = i as u8;
            buf[off] = v;
            buf[off + 1] = v;
            buf[off + 2] = v;
            // buf[off + 3] = 0 already
        }

        Ok(Self {
            buf,
            width: info.width,
            height: info.height,
            channels: 1,
            row_stride,
            data_offset,
        })
    }
}

impl StreamingEncoder for BmpStreamingEncoder {
    fn write_tile(
        &mut self,
        pixels: &[u8],
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), ImageError> {
        let ch = self.channels;
        for row in 0..h as usize {
            let img_row = y as usize + row;
            // BMP: bottom-to-top row order
            let bmp_row = self.height as usize - 1 - img_row;
            let dst_base = self.data_offset + bmp_row * self.row_stride;

            if ch == 1 {
                // Gray8: direct copy to palette-indexed position
                let src_start = row * w as usize;
                let dst_start = dst_base + x as usize;
                self.buf[dst_start..dst_start + w as usize]
                    .copy_from_slice(&pixels[src_start..src_start + w as usize]);
            } else {
                // RGB8: swap R↔B to BGR
                for col in 0..w as usize {
                    let src = (row * w as usize + col) * ch;
                    let dst = dst_base + (x as usize + col) * ch;
                    self.buf[dst] = pixels[src + 2]; // B
                    self.buf[dst + 1] = pixels[src + 1]; // G
                    self.buf[dst + 2] = pixels[src]; // R
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>, ImageError> {
        Ok(std::mem::take(&mut self.buf))
    }
}

// ─── HDR Streaming Encoder ─────────────────────────────────────────────────
//
// True streaming: writes Radiance RGBE header immediately, then converts
// pixel tiles to RGBE scanline-by-scanline. Uses a row buffer for tiles
// that don't span the full width.

pub struct HdrStreamingEncoder {
    buf: Vec<u8>,
    width: u32,
    channels: usize,
    /// Row assembly buffer: width * channels bytes per scanline row
    row_buf: Vec<u8>,
    /// How many rows are in the current tile row
    tile_h: u32,
    /// Current tile row's y position
    current_y: u32,
}

impl HdrStreamingEncoder {
    pub fn new(info: &ImageInfo) -> Result<Self, ImageError> {
        let channels = match info.format {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
            PixelFormat::Gray8 => 1,
            _ => {
                return Err(ImageError::UnsupportedFormat(
                    "HDR streaming encode requires RGB8, RGBA8, or Gray8".into(),
                ));
            }
        };

        let w = info.width as usize;
        let h = info.height as usize;

        // Estimate output size: header (~60 bytes) + w*h*4 RGBE bytes
        let mut buf = Vec::with_capacity(64 + w * h * 4);

        // Write Radiance HDR header
        buf.extend_from_slice(b"#?RADIANCE\n");
        buf.extend_from_slice(b"FORMAT=32-bit_rle_rgbe\n");
        buf.extend_from_slice(b"\n");
        let res = format!("-Y {} +X {}\n", info.height, info.width);
        buf.extend_from_slice(res.as_bytes());

        Ok(Self {
            buf,
            width: info.width,
            channels,
            row_buf: Vec::new(),
            tile_h: 0,
            current_y: u32::MAX,
        })
    }

    fn flush_rows(&mut self) {
        if self.row_buf.is_empty() || self.tile_h == 0 {
            return;
        }
        let w = self.width as usize;
        let ch = self.channels;
        for row in 0..self.tile_h as usize {
            for x in 0..w {
                let idx = (row * w + x) * ch;
                let (r, g, b) = match ch {
                    3 => (
                        self.row_buf[idx] as f32 / 255.0,
                        self.row_buf[idx + 1] as f32 / 255.0,
                        self.row_buf[idx + 2] as f32 / 255.0,
                    ),
                    4 => (
                        self.row_buf[idx] as f32 / 255.0,
                        self.row_buf[idx + 1] as f32 / 255.0,
                        self.row_buf[idx + 2] as f32 / 255.0,
                    ),
                    1 => {
                        let v = self.row_buf[idx] as f32 / 255.0;
                        (v, v, v)
                    }
                    _ => unreachable!(),
                };
                self.buf.extend_from_slice(&to_rgbe(r, g, b));
            }
        }
        self.row_buf.clear();
        self.tile_h = 0;
    }
}

impl StreamingEncoder for HdrStreamingEncoder {
    fn write_tile(
        &mut self,
        pixels: &[u8],
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), ImageError> {
        // New tile row — flush previous
        if y != self.current_y {
            self.flush_rows();
            self.current_y = y;
            self.tile_h = h;
            let row_bytes = self.width as usize * self.channels * h as usize;
            self.row_buf.resize(row_bytes, 0);
        }

        // Copy tile pixels into row buffer at correct column offset
        let ch = self.channels;
        let full_w = self.width as usize;
        for row in 0..h as usize {
            let src_start = row * w as usize * ch;
            let dst_start = (row * full_w + x as usize) * ch;
            let len = w as usize * ch;
            self.row_buf[dst_start..dst_start + len]
                .copy_from_slice(&pixels[src_start..src_start + len]);
        }

        // Flush if this tile completes the row
        if x + w >= self.width {
            self.flush_rows();
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>, ImageError> {
        self.flush_rows();
        Ok(std::mem::take(&mut self.buf))
    }
}

/// Convert linear RGB to RGBE (shared exponent) — matches hdr.rs::to_rgbe.
fn to_rgbe(r: f32, g: f32, b: f32) -> [u8; 4] {
    let max_val = r.max(g).max(b);
    if max_val < 1e-32 {
        return [0, 0, 0, 0];
    }
    let e = max_val.log2().ceil() as i32;
    let scale = (256.0 / (2.0f32.powi(e))).min(255.0);
    [
        (r * scale) as u8,
        (g * scale) as u8,
        (b * scale) as u8,
        (e + 128) as u8,
    ]
}

// ─── FITS Streaming Encoder ────────────────────────────────────────────────
//
// True streaming: pre-allocates header + data area, writes pixel bytes at
// known offsets. FITS uses big-endian row-major order (top-to-bottom).

pub struct FitsStreamingEncoder {
    buf: Vec<u8>,
    width: u32,
    data_offset: usize,
    is_rgb: bool,
}

impl FitsStreamingEncoder {
    pub fn new(info: &ImageInfo) -> Result<Self, ImageError> {
        let is_rgb = match info.format {
            PixelFormat::Gray8 => false,
            PixelFormat::Rgb8 => true,
            other => {
                return Err(ImageError::UnsupportedFormat(format!(
                    "FITS streaming encode does not support {other:?}"
                )));
            }
        };

        let w = info.width as usize;
        let h = info.height as usize;
        let pixel_count = w * h; // grayscale pixel count (RGB → luma)
        let header = Self::build_header(info.width, info.height);
        let data_offset = header.len();
        let padded_data = Self::pad_to_block(pixel_count);

        let mut buf = Vec::with_capacity(data_offset + padded_data);
        buf.extend_from_slice(&header);
        buf.resize(data_offset + padded_data, 0);

        Ok(Self {
            buf,
            width: info.width,
            data_offset,
            is_rgb,
        })
    }

    fn build_header(width: u32, height: u32) -> Vec<u8> {
        let mut header = Vec::with_capacity(2880);
        Self::write_card(&mut header, "SIMPLE", "T");
        Self::write_card(&mut header, "BITPIX", "8");
        Self::write_card(&mut header, "NAXIS", "2");
        Self::write_card(&mut header, "NAXIS1", &width.to_string());
        Self::write_card(&mut header, "NAXIS2", &height.to_string());

        // END card
        let mut end_card = [b' '; 80];
        end_card[..3].copy_from_slice(b"END");
        header.extend_from_slice(&end_card);

        // Pad to 2880-byte boundary
        let remainder = header.len() % 2880;
        if remainder != 0 {
            header.extend(std::iter::repeat_n(0u8, 2880 - remainder));
        }
        header
    }

    fn write_card(buf: &mut Vec<u8>, keyword: &str, value: &str) {
        let mut card = [b' '; 80];
        let kw = keyword.as_bytes();
        card[..kw.len().min(8)].copy_from_slice(&kw[..kw.len().min(8)]);
        card[8] = b'=';
        card[9] = b' ';
        let val_bytes = value.as_bytes();
        let start = 30usize.saturating_sub(val_bytes.len()).max(10);
        card[start..start + val_bytes.len().min(70)]
            .copy_from_slice(&val_bytes[..val_bytes.len().min(70)]);
        buf.extend_from_slice(&card);
    }

    fn pad_to_block(size: usize) -> usize {
        size.div_ceil(2880) * 2880
    }
}

impl StreamingEncoder for FitsStreamingEncoder {
    fn write_tile(
        &mut self,
        pixels: &[u8],
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), ImageError> {
        let full_w = self.width as usize;
        for row in 0..h as usize {
            let img_row = y as usize + row;
            let dst_base = self.data_offset + img_row * full_w;

            if self.is_rgb {
                // RGB → luma conversion
                for col in 0..w as usize {
                    let src = (row * w as usize + col) * 3;
                    let luma = (0.299 * pixels[src] as f64
                        + 0.587 * pixels[src + 1] as f64
                        + 0.114 * pixels[src + 2] as f64)
                        .round() as u8;
                    self.buf[dst_base + x as usize + col] = luma;
                }
            } else {
                // Gray8: direct copy
                let src_start = row * w as usize;
                let dst_start = dst_base + x as usize;
                self.buf[dst_start..dst_start + w as usize]
                    .copy_from_slice(&pixels[src_start..src_start + w as usize]);
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>, ImageError> {
        Ok(std::mem::take(&mut self.buf))
    }
}

// ─── QOI Streaming Encoder (Buffered) ──────────────────────────────────────
//
// Buffers all tiles into a pixel buffer, then calls the rasmcore-qoi encoder
// in finish(). Provides API consistency but no memory benefit.

#[cfg(feature = "native-qoi")]
pub struct QoiStreamingEncoder {
    pixel_buf: Vec<u8>,
    width: u32,
    channels: usize,
    info: ImageInfo,
}

#[cfg(feature = "native-qoi")]
impl QoiStreamingEncoder {
    pub fn new(info: &ImageInfo) -> Result<Self, ImageError> {
        let channels = match info.format {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
            other => {
                return Err(ImageError::UnsupportedFormat(format!(
                    "QOI streaming encode does not support {other:?}"
                )));
            }
        };
        let size = info.width as usize * info.height as usize * channels;
        Ok(Self {
            pixel_buf: vec![0u8; size],
            width: info.width,
            channels,
            info: info.clone(),
        })
    }
}

#[cfg(feature = "native-qoi")]
impl StreamingEncoder for QoiStreamingEncoder {
    fn write_tile(
        &mut self,
        pixels: &[u8],
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), ImageError> {
        let ch = self.channels;
        let full_w = self.width as usize;
        for row in 0..h as usize {
            let src_start = row * w as usize * ch;
            let dst_start = ((y as usize + row) * full_w + x as usize) * ch;
            let len = w as usize * ch;
            self.pixel_buf[dst_start..dst_start + len]
                .copy_from_slice(&pixels[src_start..src_start + len]);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>, ImageError> {
        super::native_trivial::encode_qoi(&self.pixel_buf, &self.info)
    }
}

// ─── TIFF Streaming Encoder (Buffered) ─────────────────────────────────────
//
// Buffers all tiles into a pixel buffer, then uses the tiff crate in
// finish(). The tiff crate's write_data() requires the complete buffer.
// API consistency only — no memory benefit over the stitch path.

pub struct TiffStreamingEncoder {
    pixel_buf: Vec<u8>,
    width: u32,
    bpp: usize,
    info: ImageInfo,
    config: super::tiff::TiffEncodeConfig,
}

impl TiffStreamingEncoder {
    pub fn new(info: &ImageInfo, config: &super::tiff::TiffEncodeConfig) -> Result<Self, ImageError> {
        let bpp = match info.format {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
            PixelFormat::Gray8 => 1,
            PixelFormat::Gray16 => 2,
            PixelFormat::Rgb16 => 6,
            PixelFormat::Rgba16 => 8,
            PixelFormat::Cmyk8 => 4,
            other => {
                return Err(ImageError::UnsupportedFormat(format!(
                    "TIFF streaming encode does not support {other:?}"
                )));
            }
        };
        let size = info.width as usize * info.height as usize * bpp;
        Ok(Self {
            pixel_buf: vec![0u8; size],
            width: info.width,
            bpp,
            info: info.clone(),
            config: config.clone(),
        })
    }
}

impl StreamingEncoder for TiffStreamingEncoder {
    fn write_tile(
        &mut self,
        pixels: &[u8],
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), ImageError> {
        let full_w = self.width as usize;
        let bpp = self.bpp;
        for row in 0..h as usize {
            let src_start = row * w as usize * bpp;
            let dst_start = ((y as usize + row) * full_w + x as usize) * bpp;
            let len = w as usize * bpp;
            self.pixel_buf[dst_start..dst_start + len]
                .copy_from_slice(&pixels[src_start..src_start + len]);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>, ImageError> {
        let pixels = std::mem::take(&mut self.pixel_buf);
        super::tiff::encode(&pixels, &self.info, &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo};

    fn make_rgb8(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    fn make_gray8(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    /// Feed a complete pixel buffer as a single tile.
    fn encode_single_tile(
        enc: &mut dyn StreamingEncoder,
        pixels: &[u8],
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        enc.write_tile(pixels, 0, 0, w, h).unwrap();
        enc.finish().unwrap()
    }

    /// Feed pixels as a grid of small tiles (tile_size x tile_size).
    fn encode_tiled(
        enc: &mut dyn StreamingEncoder,
        pixels: &[u8],
        w: u32,
        h: u32,
        ch: u32,
        tile_size: u32,
    ) -> Vec<u8> {
        let mut y = 0u32;
        while y < h {
            let th = tile_size.min(h - y);
            let mut x = 0u32;
            while x < w {
                let tw = tile_size.min(w - x);
                // Extract tile from full pixel buffer
                let mut tile = Vec::with_capacity((tw * th * ch) as usize);
                for row in 0..th {
                    let src = ((y + row) * w + x) as usize * ch as usize;
                    let len = tw as usize * ch as usize;
                    tile.extend_from_slice(&pixels[src..src + len]);
                }
                enc.write_tile(&tile, x, y, tw, th).unwrap();
                x += tile_size;
            }
            y += tile_size;
        }
        enc.finish().unwrap()
    }

    // ── BMP parity ─────────────────────────────────────────────────────

    #[test]
    fn bmp_streaming_matches_buffer_rgb() {
        let (pixels, info) = make_rgb8(17, 13); // odd dimensions to test padding
        let buffer_out = super::super::native_trivial::encode_bmp(&pixels, &info).unwrap();

        let mut enc = BmpStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_single_tile(&mut enc, &pixels, info.width, info.height);
        assert_eq!(buffer_out, stream_out, "BMP streaming must match buffer encoding");
    }

    #[test]
    fn bmp_streaming_tiled_matches_buffer() {
        let (pixels, info) = make_rgb8(40, 30);
        let buffer_out = super::super::native_trivial::encode_bmp(&pixels, &info).unwrap();

        let mut enc = BmpStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_tiled(&mut enc, &pixels, 40, 30, 3, 7);
        assert_eq!(buffer_out, stream_out, "BMP tiled streaming must match buffer");
    }

    #[test]
    fn bmp_streaming_matches_buffer_gray() {
        let (pixels, info) = make_gray8(17, 13);
        let buffer_out = super::super::native_trivial::encode_bmp(&pixels, &info).unwrap();

        let mut enc = BmpStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_single_tile(&mut enc, &pixels, info.width, info.height);
        assert_eq!(buffer_out, stream_out, "BMP gray streaming must match buffer");
    }

    // ── HDR parity ─────────────────────────────────────────────────────

    #[test]
    fn hdr_streaming_matches_buffer() {
        let (pixels, info) = make_rgb8(16, 16);
        let buffer_out =
            super::super::hdr::encode_pixels(&pixels, &info, &super::super::hdr::HdrEncodeConfig)
                .unwrap();

        let mut enc = HdrStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_single_tile(&mut enc, &pixels, info.width, info.height);
        assert_eq!(buffer_out, stream_out, "HDR streaming must match buffer encoding");
    }

    #[test]
    fn hdr_streaming_tiled_matches_buffer() {
        let (pixels, info) = make_rgb8(40, 30);
        let buffer_out =
            super::super::hdr::encode_pixels(&pixels, &info, &super::super::hdr::HdrEncodeConfig)
                .unwrap();

        let mut enc = HdrStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_tiled(&mut enc, &pixels, 40, 30, 3, 11);
        assert_eq!(buffer_out, stream_out, "HDR tiled streaming must match buffer");
    }

    // ── FITS parity ────────────────────────────────────────────────────

    #[test]
    fn fits_streaming_matches_buffer_gray() {
        let (pixels, info) = make_gray8(32, 32);
        let buffer_out = super::super::fits::encode_pixels(&pixels, &info).unwrap();

        let mut enc = FitsStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_single_tile(&mut enc, &pixels, info.width, info.height);
        assert_eq!(buffer_out, stream_out, "FITS streaming must match buffer");
    }

    #[test]
    fn fits_streaming_tiled_matches_buffer() {
        let (pixels, info) = make_gray8(40, 30);
        let buffer_out = super::super::fits::encode_pixels(&pixels, &info).unwrap();

        let mut enc = FitsStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_tiled(&mut enc, &pixels, 40, 30, 1, 9);
        assert_eq!(buffer_out, stream_out, "FITS tiled streaming must match buffer");
    }

    #[test]
    fn fits_streaming_rgb_matches_buffer() {
        let (pixels, info) = make_rgb8(16, 16);
        let buffer_out = super::super::fits::encode_pixels(&pixels, &info).unwrap();

        let mut enc = FitsStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_single_tile(&mut enc, &pixels, info.width, info.height);
        assert_eq!(buffer_out, stream_out, "FITS RGB streaming must match buffer");
    }

    // ── QOI parity ─────────────────────────────────────────────────────

    #[cfg(feature = "native-qoi")]
    #[test]
    fn qoi_streaming_matches_buffer() {
        let (pixels, info) = make_rgb8(16, 16);
        let buffer_out = super::super::native_trivial::encode_qoi(&pixels, &info).unwrap();

        let mut enc = QoiStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_single_tile(&mut enc, &pixels, info.width, info.height);
        assert_eq!(buffer_out, stream_out, "QOI streaming must match buffer");
    }

    #[cfg(feature = "native-qoi")]
    #[test]
    fn qoi_streaming_tiled_matches_buffer() {
        let (pixels, info) = make_rgb8(40, 30);
        let buffer_out = super::super::native_trivial::encode_qoi(&pixels, &info).unwrap();

        let mut enc = QoiStreamingEncoder::new(&info).unwrap();
        let stream_out = encode_tiled(&mut enc, &pixels, 40, 30, 3, 13);
        assert_eq!(buffer_out, stream_out, "QOI tiled streaming must match buffer");
    }

    // ── TIFF parity ────────────────────────────────────────────────────

    #[test]
    fn tiff_streaming_matches_buffer() {
        let (pixels, info) = make_rgb8(16, 16);
        let config = super::super::tiff::TiffEncodeConfig::default();
        let buffer_out = super::super::tiff::encode(&pixels, &info, &config).unwrap();

        let mut enc = TiffStreamingEncoder::new(&info, &config).unwrap();
        let stream_out = encode_single_tile(&mut enc, &pixels, info.width, info.height);
        assert_eq!(buffer_out, stream_out, "TIFF streaming must match buffer");
    }

    #[test]
    fn tiff_streaming_tiled_matches_buffer() {
        let (pixels, info) = make_rgb8(40, 30);
        let config = super::super::tiff::TiffEncodeConfig::default();
        let buffer_out = super::super::tiff::encode(&pixels, &info, &config).unwrap();

        let mut enc = TiffStreamingEncoder::new(&info, &config).unwrap();
        let stream_out = encode_tiled(&mut enc, &pixels, 40, 30, 3, 7);
        assert_eq!(buffer_out, stream_out, "TIFF tiled streaming must match buffer");
    }
}

/// Create a streaming encoder for the given format, if supported.
///
/// Returns `None` for formats that don't support streaming (PNG, JPEG,
/// WebP, GIF, AVIF, etc.).
pub fn create_streaming_encoder(
    format: &str,
    info: &ImageInfo,
) -> Option<Box<dyn StreamingEncoder>> {
    match format {
        "bmp" => BmpStreamingEncoder::new(info).ok().map(|e| Box::new(e) as _),
        "hdr" => HdrStreamingEncoder::new(info).ok().map(|e| Box::new(e) as _),
        "fits" | "fit" => FitsStreamingEncoder::new(info).ok().map(|e| Box::new(e) as _),
        #[cfg(feature = "native-qoi")]
        "qoi" => QoiStreamingEncoder::new(info).ok().map(|e| Box::new(e) as _),
        "tiff" | "tif" => {
            let config = super::tiff::TiffEncodeConfig::default();
            TiffStreamingEncoder::new(info, &config).ok().map(|e| Box::new(e) as _)
        }
        _ => None,
    }
}
