//! Filter: mirror_kaleidoscope (category: effect)
//!
//! Mirror/kaleidoscope effect: reflects the image along an axis. With
//! segment count > 2, creates a kaleidoscope by dividing the image into
//! angular segments and mirroring alternating ones.
//! Reference: PicsArt/Snapseed mirror/kaleidoscope effect.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for the mirror/kaleidoscope effect.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct MirrorKaleidoscopeParams {
    /// Number of angular segments (2 = simple mirror, 4+ = kaleidoscope)
    #[param(min = 2, max = 24, step = 2, default = 2)]
    pub segments: u32,
    /// Rotation angle of the mirror axis in degrees
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 0.0, hint = "rc.angle_deg")]
    pub angle: f32,
    /// Mirror mode: 0 = horizontal, 1 = vertical, 2 = angular/kaleidoscope
    #[param(min = 0, max = 2, step = 1, default = 0)]
    pub mode: u32,
}

#[rasmcore_macros::register_filter(
    name = "mirror_kaleidoscope",
    category = "effect",
    reference = "mirror and angular kaleidoscope"
)]
pub fn mirror_kaleidoscope(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &MirrorKaleidoscopeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            mirror_kaleidoscope(r, &mut u, i8, config)
        });
    }
    if is_f32(info.format) {
        return process_via_standard(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            mirror_kaleidoscope(r, &mut u, i8, config)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let mode = config.mode;

    match mode {
        0 => mirror_horizontal(pixels, w, h, ch),
        1 => mirror_vertical(pixels, w, h, ch),
        _ => kaleidoscope(pixels, w, h, ch, config.segments.max(2), config.angle),
    }
}

/// Simple horizontal mirror: left half reflected to right.
fn mirror_horizontal(pixels: &[u8], w: usize, h: usize, ch: usize) -> Result<Vec<u8>, ImageError> {
    let mut out = pixels.to_vec();
    let mid = w / 2;
    for y in 0..h {
        for x in mid..w {
            let mirror_x = w - 1 - x;
            let dst = (y * w + x) * ch;
            let src = (y * w + mirror_x) * ch;
            out[dst..dst + ch].copy_from_slice(&pixels[src..src + ch]);
        }
    }
    Ok(out)
}

/// Simple vertical mirror: top half reflected to bottom.
fn mirror_vertical(pixels: &[u8], w: usize, h: usize, ch: usize) -> Result<Vec<u8>, ImageError> {
    let mut out = pixels.to_vec();
    let mid = h / 2;
    for y in mid..h {
        let mirror_y = h - 1 - y;
        let dst_row = y * w * ch;
        let src_row = mirror_y * w * ch;
        out[dst_row..dst_row + w * ch].copy_from_slice(&pixels[src_row..src_row + w * ch]);
    }
    Ok(out)
}

/// Angular kaleidoscope: divide into N angular segments, mirror alternating ones.
fn kaleidoscope(
    pixels: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    segments: u32,
    angle_offset: f32,
) -> Result<Vec<u8>, ImageError> {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let seg_angle = 2.0 * std::f32::consts::PI / segments as f32;
    let offset_rad = angle_offset.to_radians();

    let mut out = vec![0u8; pixels.len()];

    for y in 0..h {
        let dy = y as f32 - cy;
        for x in 0..w {
            let dx = x as f32 - cx;

            // Convert to polar coordinates
            let mut theta = dy.atan2(dx) - offset_rad;
            let radius = (dx * dx + dy * dy).sqrt();

            // Normalize theta to [0, 2*PI)
            while theta < 0.0 {
                theta += 2.0 * std::f32::consts::PI;
            }
            while theta >= 2.0 * std::f32::consts::PI {
                theta -= 2.0 * std::f32::consts::PI;
            }

            // Find which segment this pixel is in
            let seg_idx = (theta / seg_angle) as u32;
            let seg_start = seg_idx as f32 * seg_angle;
            let local_angle = theta - seg_start;

            // Mirror alternating segments
            let mapped_angle = if seg_idx % 2 == 0 {
                local_angle + offset_rad
            } else {
                (seg_angle - local_angle) + offset_rad
            };

            // Convert back to Cartesian
            let src_x = (cx + radius * mapped_angle.cos()).round() as isize;
            let src_y = (cy + radius * mapped_angle.sin()).round() as isize;

            let dst_idx = (y * w + x) * ch;

            // Clamp to bounds
            let sx = src_x.clamp(0, w as isize - 1) as usize;
            let sy = src_y.clamp(0, h as isize - 1) as usize;
            let src_idx = (sy * w + sx) * ch;

            out[dst_idx..dst_idx + ch].copy_from_slice(&pixels[src_idx..src_idx + ch]);
        }
    }

    Ok(out)
}
