//! Filter: perspective_correct (category: advanced)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Automatic perspective correction — detects dominant lines and rectifies.
///
/// Pipeline: Canny edges → Hough lines → classify H/V → estimate vanishing
/// points → compute rectifying homography → warp.
///
/// - `strength`: correction strength 0.0 (none) to 1.0 (full correction)
///
/// The output has the same dimensions and format as the input.

/// Parameters for perspective_correct.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "perspective_correct", category = "advanced", group = "perspective", variant = "correct", reference = "automatic perspective rectification")]
pub struct PerspectiveCorrectParams {
    /// Correction strength (0=none, 1=full)
    #[param(min = 0.0, max = 2.0, step = 0.1, default = 1.0)]
    pub strength: f32,
}

impl CpuFilter for PerspectiveCorrectParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let strength = self.strength;

    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            self.compute(r, &mut u, i8)
        });
    }

    if strength <= 0.0 {
        return Ok(pixels.to_vec());
    }

    let w = info.width as i32;
    let h = info.height as i32;

    // Step 1: Edge detection
    let edge_map = canny(
        pixels,
        info,
        &CannyParams {
            low_threshold: 50.0,
            high_threshold: 150.0,
        },
    )?;
    let edge_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };

    // Step 2: Line detection
    let min_dim = w.min(h) as f32;
    let min_length = ((min_dim * 0.1).max(30.0)) as i32;
    let lines = hough_lines_p(
        &edge_map,
        &edge_info,
        1.0,                          // rho resolution
        std::f32::consts::PI / 180.0, // theta resolution (1 degree)
        (min_dim * 0.15) as i32,      // threshold scales with image size
        min_length,
        (min_length as f32 * 0.3) as i32, // max gap
        0,                                // default seed
    )?;

    if lines.len() < 4 {
        return Ok(pixels.to_vec()); // Not enough lines to correct
    }

    // Step 3: Classify lines as near-horizontal or near-vertical
    let mut h_lines = Vec::new();
    let mut v_lines = Vec::new();
    let angle_threshold = 20.0f32.to_radians();

    for line in &lines {
        let dx = (line.x2 - line.x1) as f32;
        let dy = (line.y2 - line.y1) as f32;
        let angle = dy.atan2(dx).abs();
        let length = (dx * dx + dy * dy).sqrt();

        if angle < angle_threshold || angle > (std::f32::consts::PI - angle_threshold) {
            h_lines.push((*line, length));
        } else if (angle - std::f32::consts::FRAC_PI_2).abs() < angle_threshold {
            v_lines.push((*line, length));
        }
    }

    if h_lines.len() < 2 && v_lines.len() < 2 {
        return Ok(pixels.to_vec());
    }

    // Step 4: Estimate vanishing points
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;

    let mut angle_x = 0.0f32;
    let mut angle_y = 0.0f32;

    if v_lines.len() >= 2
        && let Some(vp) = estimate_vanishing_point(&v_lines)
    {
        let dx = vp.0 - cx;
        angle_x = (dx / (h as f32 * 2.0)).atan() * strength;
    }

    if h_lines.len() >= 2
        && let Some(vp) = estimate_vanishing_point(&h_lines)
    {
        let dy = vp.1 - cy;
        angle_y = (dy / (w as f32 * 2.0)).atan() * strength;
    }

    // Step 5: Build rectifying homography
    let hw = w as f32 / 2.0;
    let hh = h as f32 / 2.0;
    let shift_top_x = -angle_x * hh;
    let shift_bot_x = angle_x * hh;
    let shift_left_y = -angle_y * hw;
    let shift_right_y = angle_y * hw;

    let src_corners = [
        (0.0f32, 0.0f32),
        (w as f32, 0.0),
        (w as f32, h as f32),
        (0.0, h as f32),
    ];

    let dst_corners = [
        (shift_top_x + shift_left_y, shift_left_y + shift_top_x),
        (
            w as f32 - shift_top_x + shift_right_y,
            shift_right_y + shift_top_x,
        ),
        (
            w as f32 - shift_bot_x + shift_right_y,
            h as f32 - shift_right_y - shift_bot_x,
        ),
        (
            shift_bot_x + shift_left_y,
            h as f32 - shift_left_y - shift_bot_x,
        ),
    ];

    let h_mat = match solve_homography_4pt(
        &[
            dst_corners[0],
            dst_corners[1],
            dst_corners[2],
            dst_corners[3],
        ],
        &[
            src_corners[0],
            src_corners[1],
            src_corners[2],
            src_corners[3],
        ],
    ) {
        Some(h) => h,
        None => return Ok(pixels.to_vec()),
    };

    {
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.to_vec());
        perspective_warp(
            r,
            &mut u,
            info,
            &h_mat,
            &PerspectiveWarpParams {
                out_width: info.width,
                out_height: info.height,
            },
        )
    }
}
}

