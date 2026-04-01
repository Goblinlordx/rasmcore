//! Filter: hough_lines (category: analysis)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct HoughLinesParams {
    pub threshold: u32,
    pub min_length: u32,
    pub max_gap: u32,
}

#[rasmcore_macros::register_filter(
    name = "hough_lines",
    category = "analysis",
    group = "analysis",
    variant = "hough_lines",
    reference = "probabilistic Hough transform (Matas et al. 2000)"
)]
pub fn hough_lines_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &HoughLinesParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let threshold = config.threshold;
    let min_length = config.min_length;
    let max_gap = config.max_gap;

    let lines = hough_lines_p(
        pixels,
        info,
        1.0,                          // rho = 1 pixel
        std::f32::consts::PI / 180.0, // theta = 1 degree
        threshold as i32,
        min_length as i32,
        max_gap as i32,
        12345, // fixed seed for reproducibility
    )?;
    // Render detected lines onto a blank Gray8 canvas
    let w = info.width as usize;
    let h = info.height as usize;
    let mut out = vec![0u8; w * h];
    for seg in &lines {
        // Bresenham line drawing
        let (mut x, mut y) = (seg.x1, seg.y1);
        let dx = (seg.x2 - seg.x1).abs();
        let dy = -(seg.y2 - seg.y1).abs();
        let sx = if seg.x1 < seg.x2 { 1 } else { -1 };
        let sy = if seg.y1 < seg.y2 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
                out[y as usize * w + x as usize] = 255;
            }
            if x == seg.x2 && y == seg.y2 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }
    Ok(out)
}
