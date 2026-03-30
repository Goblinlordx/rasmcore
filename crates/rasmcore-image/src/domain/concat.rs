//! Image concatenation — horizontal, vertical, and grid layouts.
//!
//! Matches libvips `arrayjoin` and ImageMagick `append` / `montage` operations.

use super::error::ImageError;
use super::types::{DecodedImage, ImageInfo, PixelFormat};

/// Bytes per pixel for supported formats. Mirrors transform.rs helper.
fn bpp(format: PixelFormat) -> Result<usize, ImageError> {
    match format {
        PixelFormat::Rgb8 => Ok(3),
        PixelFormat::Rgba8 => Ok(4),
        PixelFormat::Gray8 => Ok(1),
        PixelFormat::Gray16 => Ok(2),
        PixelFormat::Rgb16 => Ok(6),
        PixelFormat::Rgba16 => Ok(8),
        _ => Err(ImageError::UnsupportedFormat(format!(
            "concat: {format:?} not supported"
        ))),
    }
}

/// Fill a row-major pixel buffer region with a solid color.
fn fill_rect(
    buf: &mut [u8],
    buf_w: usize,
    bpp: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: &[u8],
) {
    for row in y..y + h {
        for col in x..x + w {
            let offset = (row * buf_w + col) * bpp;
            for c in 0..bpp {
                buf[offset + c] = color.get(c).copied().unwrap_or(0);
            }
        }
    }
}

/// Blit (copy) a source image into a destination buffer at the given offset.
fn blit(
    dst: &mut [u8],
    dst_w: usize,
    src: &[u8],
    src_w: usize,
    src_h: usize,
    bpp: usize,
    dst_x: usize,
    dst_y: usize,
) {
    let src_stride = src_w * bpp;
    let dst_stride = dst_w * bpp;
    for row in 0..src_h {
        let src_start = row * src_stride;
        let dst_start = (dst_y + row) * dst_stride + dst_x * bpp;
        dst[dst_start..dst_start + src_stride]
            .copy_from_slice(&src[src_start..src_start + src_stride]);
    }
}

/// Concatenate images horizontally (left to right).
///
/// All images must share the same pixel format. Images of different heights are
/// centered vertically with `bg_color` fill. `gap` adds spacing between images.
pub fn concat_horizontal(
    images: &[(&[u8], &ImageInfo)],
    gap: u32,
    bg_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    if images.is_empty() {
        return Err(ImageError::InvalidInput("concat: no images provided".into()));
    }

    let format = images[0].1.format;
    let color_space = images[0].1.color_space;
    let bytes = bpp(format)?;

    for (i, (_, info)) in images.iter().enumerate() {
        if info.format != format {
            return Err(ImageError::InvalidInput(format!(
                "concat: image {} has format {:?}, expected {:?}",
                i, info.format, format
            )));
        }
    }

    let max_h = images.iter().map(|(_, info)| info.height).max().unwrap_or(0) as usize;
    let total_w: usize = images.iter().map(|(_, info)| info.width as usize).sum::<usize>()
        + gap as usize * images.len().saturating_sub(1);

    if total_w == 0 || max_h == 0 {
        return Err(ImageError::InvalidInput("concat: zero-size output".into()));
    }

    let mut output = vec![0u8; total_w * max_h * bytes];

    // Fill with background
    fill_rect(&mut output, total_w, bytes, 0, 0, total_w, max_h, bg_color);

    let mut x_offset = 0usize;
    for (pixels, info) in images {
        let w = info.width as usize;
        let h = info.height as usize;
        // Center vertically
        let y_offset = (max_h - h) / 2;
        blit(&mut output, total_w, pixels, w, h, bytes, x_offset, y_offset);
        x_offset += w + gap as usize;
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: total_w as u32,
            height: max_h as u32,
            format,
            color_space,
        },
        icc_profile: None,
    })
}

/// Concatenate images vertically (top to bottom).
///
/// All images must share the same pixel format. Images of different widths are
/// centered horizontally with `bg_color` fill. `gap` adds spacing between images.
pub fn concat_vertical(
    images: &[(&[u8], &ImageInfo)],
    gap: u32,
    bg_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    if images.is_empty() {
        return Err(ImageError::InvalidInput("concat: no images provided".into()));
    }

    let format = images[0].1.format;
    let color_space = images[0].1.color_space;
    let bytes = bpp(format)?;

    for (i, (_, info)) in images.iter().enumerate() {
        if info.format != format {
            return Err(ImageError::InvalidInput(format!(
                "concat: image {} has format {:?}, expected {:?}",
                i, info.format, format
            )));
        }
    }

    let max_w = images.iter().map(|(_, info)| info.width).max().unwrap_or(0) as usize;
    let total_h: usize = images.iter().map(|(_, info)| info.height as usize).sum::<usize>()
        + gap as usize * images.len().saturating_sub(1);

    if max_w == 0 || total_h == 0 {
        return Err(ImageError::InvalidInput("concat: zero-size output".into()));
    }

    let mut output = vec![0u8; max_w * total_h * bytes];
    fill_rect(&mut output, max_w, bytes, 0, 0, max_w, total_h, bg_color);

    let mut y_offset = 0usize;
    for (pixels, info) in images {
        let w = info.width as usize;
        let h = info.height as usize;
        let x_offset = (max_w - w) / 2;
        blit(&mut output, max_w, pixels, w, h, bytes, x_offset, y_offset);
        y_offset += h + gap as usize;
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: max_w as u32,
            height: total_h as u32,
            format,
            color_space,
        },
        icc_profile: None,
    })
}

/// Arrange images in a grid with the given number of columns.
///
/// All images must share the same pixel format. Each cell is sized to the maximum
/// width and height across all images. Images are centered within their cells.
/// `gap` adds spacing between cells. `bg_color` fills empty space.
pub fn concat_grid(
    images: &[(&[u8], &ImageInfo)],
    columns: u32,
    gap: u32,
    bg_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    if images.is_empty() {
        return Err(ImageError::InvalidInput("concat_grid: no images provided".into()));
    }
    if columns == 0 {
        return Err(ImageError::InvalidParameters(
            "concat_grid: columns must be > 0".into(),
        ));
    }

    let format = images[0].1.format;
    let color_space = images[0].1.color_space;
    let bytes = bpp(format)?;

    for (i, (_, info)) in images.iter().enumerate() {
        if info.format != format {
            return Err(ImageError::InvalidInput(format!(
                "concat_grid: image {} has format {:?}, expected {:?}",
                i, info.format, format
            )));
        }
    }

    let cols = columns as usize;
    let rows = (images.len() + cols - 1) / cols;

    let cell_w = images.iter().map(|(_, info)| info.width as usize).max().unwrap_or(0);
    let cell_h = images.iter().map(|(_, info)| info.height as usize).max().unwrap_or(0);

    let total_w = cell_w * cols + gap as usize * cols.saturating_sub(1);
    let total_h = cell_h * rows + gap as usize * rows.saturating_sub(1);

    if total_w == 0 || total_h == 0 {
        return Err(ImageError::InvalidInput("concat_grid: zero-size output".into()));
    }

    let mut output = vec![0u8; total_w * total_h * bytes];
    fill_rect(&mut output, total_w, bytes, 0, 0, total_w, total_h, bg_color);

    for (idx, (pixels, info)) in images.iter().enumerate() {
        let col = idx % cols;
        let row = idx / cols;
        let w = info.width as usize;
        let h = info.height as usize;

        // Cell top-left
        let cell_x = col * (cell_w + gap as usize);
        let cell_y = row * (cell_h + gap as usize);

        // Center image within cell
        let x_offset = cell_x + (cell_w - w) / 2;
        let y_offset = cell_y + (cell_h - h) / 2;

        blit(&mut output, total_w, pixels, w, h, bytes, x_offset, y_offset);
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: total_w as u32,
            height: total_h as u32,
            format,
            color_space,
        },
        icc_profile: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn make_solid(w: u32, h: u32, r: u8, g: u8, b: u8) -> (Vec<u8>, ImageInfo) {
        let pixels = vec![r, g, b].repeat((w * h) as usize);
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn horizontal_same_size() {
        let (p1, i1) = make_solid(10, 20, 255, 0, 0);
        let (p2, i2) = make_solid(10, 20, 0, 255, 0);
        let (p3, i3) = make_solid(10, 20, 0, 0, 255);
        let images = vec![
            (p1.as_slice(), &i1),
            (p2.as_slice(), &i2),
            (p3.as_slice(), &i3),
        ];
        let result = concat_horizontal(&images, 0, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 30);
        assert_eq!(result.info.height, 20);
        assert_eq!(result.pixels.len(), 30 * 20 * 3);
        // First pixel should be red
        assert_eq!(&result.pixels[0..3], &[255, 0, 0]);
        // Pixel at x=10 should be green
        assert_eq!(&result.pixels[10 * 3..10 * 3 + 3], &[0, 255, 0]);
        // Pixel at x=20 should be blue
        assert_eq!(&result.pixels[20 * 3..20 * 3 + 3], &[0, 0, 255]);
    }

    #[test]
    fn vertical_same_size() {
        let (p1, i1) = make_solid(20, 10, 255, 0, 0);
        let (p2, i2) = make_solid(20, 10, 0, 255, 0);
        let images = vec![(p1.as_slice(), &i1), (p2.as_slice(), &i2)];
        let result = concat_vertical(&images, 0, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 20);
        assert_eq!(result.info.height, 20);
        // Row 0 should be red
        assert_eq!(&result.pixels[0..3], &[255, 0, 0]);
        // Row 10 should be green
        let row10 = 10 * 20 * 3;
        assert_eq!(&result.pixels[row10..row10 + 3], &[0, 255, 0]);
    }

    #[test]
    fn horizontal_different_heights() {
        let (p1, i1) = make_solid(10, 10, 255, 0, 0);
        let (p2, i2) = make_solid(10, 20, 0, 255, 0); // taller
        let images = vec![(p1.as_slice(), &i1), (p2.as_slice(), &i2)];
        let result = concat_horizontal(&images, 0, &[128, 128, 128]).unwrap();
        assert_eq!(result.info.width, 20);
        assert_eq!(result.info.height, 20); // max height
        // Top-left corner of first image should be bg (centered, offset 5 rows down)
        assert_eq!(&result.pixels[0..3], &[128, 128, 128]);
        // Row 5, col 0 should be red (first image centered)
        let px = (5 * 20 + 0) * 3;
        assert_eq!(&result.pixels[px..px + 3], &[255, 0, 0]);
    }

    #[test]
    fn vertical_different_widths() {
        let (p1, i1) = make_solid(10, 10, 255, 0, 0);
        let (p2, i2) = make_solid(20, 10, 0, 255, 0); // wider
        let images = vec![(p1.as_slice(), &i1), (p2.as_slice(), &i2)];
        let result = concat_vertical(&images, 0, &[128, 128, 128]).unwrap();
        assert_eq!(result.info.width, 20); // max width
        assert_eq!(result.info.height, 20);
        // First row, col 0 should be bg (10px image centered in 20px width, offset=5)
        assert_eq!(&result.pixels[0..3], &[128, 128, 128]);
        // First row, col 5 should be red
        let px = 5 * 3;
        assert_eq!(&result.pixels[px..px + 3], &[255, 0, 0]);
    }

    #[test]
    fn grid_3_columns() {
        let mut images_data = Vec::new();
        for i in 0..6u8 {
            images_data.push(make_solid(10, 10, i * 40, 0, 0));
        }
        let images: Vec<(&[u8], &ImageInfo)> = images_data
            .iter()
            .map(|(p, i)| (p.as_slice(), i))
            .collect();
        let result = concat_grid(&images, 3, 0, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 30); // 3 cols × 10
        assert_eq!(result.info.height, 20); // 2 rows × 10
    }

    #[test]
    fn grid_with_gap() {
        let (p1, i1) = make_solid(10, 10, 255, 0, 0);
        let (p2, i2) = make_solid(10, 10, 0, 255, 0);
        let (p3, i3) = make_solid(10, 10, 0, 0, 255);
        let (p4, i4) = make_solid(10, 10, 255, 255, 0);
        let images = vec![
            (p1.as_slice(), &i1),
            (p2.as_slice(), &i2),
            (p3.as_slice(), &i3),
            (p4.as_slice(), &i4),
        ];
        let result = concat_grid(&images, 2, 5, &[128, 128, 128]).unwrap();
        // 2 cols × 10 + 1 gap × 5 = 25
        assert_eq!(result.info.width, 25);
        // 2 rows × 10 + 1 gap × 5 = 25
        assert_eq!(result.info.height, 25);
        // Gap pixel at (10, 0) should be bg
        let gap_px = 10 * 3;
        assert_eq!(&result.pixels[gap_px..gap_px + 3], &[128, 128, 128]);
    }

    #[test]
    fn grid_with_gap_and_bg_color() {
        let (p1, i1) = make_solid(8, 8, 255, 0, 0);
        let (p2, i2) = make_solid(8, 8, 0, 255, 0);
        let (p3, i3) = make_solid(8, 8, 0, 0, 255);
        let images = vec![
            (p1.as_slice(), &i1),
            (p2.as_slice(), &i2),
            (p3.as_slice(), &i3),
        ];
        let result = concat_grid(&images, 3, 2, &[200, 200, 200]).unwrap();
        // 3 cols × 8 + 2 gaps × 2 = 28
        assert_eq!(result.info.width, 28);
        assert_eq!(result.info.height, 8); // 1 row
        // Check gap pixel between images at x=8
        let px = 8 * 3;
        assert_eq!(&result.pixels[px..px + 3], &[200, 200, 200]);
    }

    #[test]
    fn format_mismatch_errors() {
        let (p1, i1) = make_solid(10, 10, 255, 0, 0);
        let p2 = vec![255u8; 10 * 10 * 4]; // RGBA
        let i2 = ImageInfo {
            width: 10,
            height: 10,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let images = vec![(p1.as_slice(), &i1), (p2.as_slice(), &i2)];
        assert!(concat_horizontal(&images, 0, &[0, 0, 0]).is_err());
    }
}
