//! Filter: mesh_warp (category: distortion)
//!
//! Arbitrary grid-based mesh warp with bilinear interpolation between
//! control points. The grid defines source and destination positions for
//! an M×N grid of control points. For each output pixel, the containing
//! quad in the destination grid is found and bilinear interpolation maps
//! back to source coordinates.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for mesh warp distortion.
///
/// The grid is a flat array of control points in row-major order.
/// Each point has (src_x, src_y, dst_x, dst_y) as fractions of image
/// dimensions (0.0–1.0). Grid dimensions: grid_cols × grid_rows points.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct MeshWarpParams {
    /// Number of columns in the control point grid
    #[param(min = 2, max = 64, default = 4)]
    pub grid_cols: u32,
    /// Number of rows in the control point grid
    #[param(min = 2, max = 64, default = 4)]
    pub grid_rows: u32,
    /// Grid control points as JSON: [[src_x, src_y, dst_x, dst_y], ...]
    /// Coordinates are fractions of image dimensions (0.0–1.0), row-major.
    #[param(default = "")]
    pub grid_json: String,
}

/// A single control point with source and destination positions in pixels.
#[derive(Clone, Copy)]
struct ControlPoint {
    src_x: f32,
    src_y: f32,
    dst_x: f32,
    dst_y: f32,
}

/// Parse grid points from JSON string. Each element is [src_x, src_y, dst_x, dst_y]
/// in normalized coordinates (0..1). Returns pixel coordinates.
fn parse_grid(json: &str, w: f32, h: f32, expected: usize) -> Result<Vec<ControlPoint>, ImageError> {
    // Minimal JSON array parser for [[f,f,f,f], ...]
    let trimmed = json.trim();
    if trimmed.is_empty() {
        return Err(ImageError::InvalidParameters("mesh_warp: grid_json is empty".into()));
    }

    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| ImageError::InvalidParameters("mesh_warp: grid_json must be a JSON array".into()))?;

    let mut points = Vec::with_capacity(expected);
    let mut depth = 0i32;
    let mut start = 0usize;

    for (i, ch) in inner.char_indices() {
        match ch {
            '[' => {
                if depth == 0 { start = i + 1; }
                depth += 1;
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    let segment = &inner[start..i];
                    let nums: Vec<f32> = segment
                        .split(',')
                        .map(|s| s.trim().parse::<f32>())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| ImageError::InvalidParameters(
                            format!("mesh_warp: invalid number in grid: {e}")
                        ))?;
                    if nums.len() != 4 {
                        return Err(ImageError::InvalidParameters(
                            format!("mesh_warp: each grid point must have 4 values, got {}", nums.len())
                        ));
                    }
                    points.push(ControlPoint {
                        src_x: nums[0] * w,
                        src_y: nums[1] * h,
                        dst_x: nums[2] * w,
                        dst_y: nums[3] * h,
                    });
                }
            }
            _ => {}
        }
    }

    if points.len() != expected {
        return Err(ImageError::InvalidParameters(
            format!("mesh_warp: expected {} grid points ({}x{}), got {}",
                    expected, expected, 1, points.len())
        ));
    }

    Ok(points)
}

/// Generate an identity grid (src == dst) for the given dimensions.
fn identity_grid(cols: u32, rows: u32, w: f32, h: f32) -> Vec<ControlPoint> {
    let mut points = Vec::with_capacity((cols * rows) as usize);
    for r in 0..rows {
        for c in 0..cols {
            let x = c as f32 / (cols - 1) as f32 * w;
            let y = r as f32 / (rows - 1) as f32 * h;
            points.push(ControlPoint {
                src_x: x,
                src_y: y,
                dst_x: x,
                dst_y: y,
            });
        }
    }
    points
}

/// Find which quad cell (col, row) in the destination grid contains pixel (px, py).
/// Returns (col, row) of the top-left corner of the containing quad, or None.
fn find_quad(
    grid: &[ControlPoint],
    cols: u32,
    px: f32,
    py: f32,
    rows: u32,
) -> Option<(u32, u32)> {
    // Linear scan over quads — for small grids this is fast enough
    for r in 0..rows - 1 {
        for c in 0..cols - 1 {
            let tl = &grid[(r * cols + c) as usize];
            let tr = &grid[(r * cols + c + 1) as usize];
            let bl = &grid[((r + 1) * cols + c) as usize];
            let br = &grid[((r + 1) * cols + c + 1) as usize];

            if point_in_quad(px, py, tl.dst_x, tl.dst_y, tr.dst_x, tr.dst_y,
                             br.dst_x, br.dst_y, bl.dst_x, bl.dst_y) {
                return Some((c, r));
            }
        }
    }
    None
}

/// Check if point (px, py) is inside the quad defined by 4 corners (CW or CCW).
/// Uses cross-product sign test for convex quads.
fn point_in_quad(
    px: f32, py: f32,
    x0: f32, y0: f32, x1: f32, y1: f32,
    x2: f32, y2: f32, x3: f32, y3: f32,
) -> bool {
    let cross = |ax: f32, ay: f32, bx: f32, by: f32| ax * by - ay * bx;

    let d0 = cross(x1 - x0, y1 - y0, px - x0, py - y0);
    let d1 = cross(x2 - x1, y2 - y1, px - x1, py - y1);
    let d2 = cross(x3 - x2, y3 - y2, px - x2, py - y2);
    let d3 = cross(x0 - x3, y0 - y3, px - x3, py - y3);

    let all_pos = d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0;
    let all_neg = d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0;
    all_pos || all_neg
}

/// Compute bilinear interpolation parameters (u, v) for point (px, py)
/// within the quad defined by corners tl, tr, bl, br (destination positions).
/// Then use (u, v) to interpolate in the source grid.
fn bilinear_inverse(
    px: f32, py: f32,
    tl: &ControlPoint, tr: &ControlPoint,
    bl: &ControlPoint, br: &ControlPoint,
) -> (f32, f32) {
    // Iterative Newton's method to find (u, v) such that:
    //   bilinear(u,v) = (px, py)
    // where bilinear(u,v) = (1-v)*((1-u)*tl + u*tr) + v*((1-u)*bl + u*br)
    let mut u = 0.5f32;
    let mut v = 0.5f32;

    for _ in 0..8 {
        // Forward: compute position at (u, v)
        let fx = (1.0 - v) * ((1.0 - u) * tl.dst_x + u * tr.dst_x)
               + v * ((1.0 - u) * bl.dst_x + u * br.dst_x);
        let fy = (1.0 - v) * ((1.0 - u) * tl.dst_y + u * tr.dst_y)
               + v * ((1.0 - u) * bl.dst_y + u * br.dst_y);

        let ex = px - fx;
        let ey = py - fy;

        if ex * ex + ey * ey < 0.001 {
            break;
        }

        // Jacobian df/d(u,v)
        let dfx_du = (1.0 - v) * (tr.dst_x - tl.dst_x) + v * (br.dst_x - bl.dst_x);
        let dfx_dv = (1.0 - u) * (bl.dst_x - tl.dst_x) + u * (br.dst_x - tr.dst_x);
        let dfy_du = (1.0 - v) * (tr.dst_y - tl.dst_y) + v * (br.dst_y - bl.dst_y);
        let dfy_dv = (1.0 - u) * (bl.dst_y - tl.dst_y) + u * (br.dst_y - tr.dst_y);

        let det = dfx_du * dfy_dv - dfx_dv * dfy_du;
        if det.abs() < 1e-10 {
            break;
        }
        let inv_det = 1.0 / det;

        u += (dfy_dv * ex - dfx_dv * ey) * inv_det;
        v += (dfx_du * ey - dfy_du * ex) * inv_det;

        u = u.clamp(0.0, 1.0);
        v = v.clamp(0.0, 1.0);
    }

    // Map (u, v) to source coordinates via bilinear interpolation on source grid
    let src_x = (1.0 - v) * ((1.0 - u) * tl.src_x + u * tr.src_x)
              + v * ((1.0 - u) * bl.src_x + u * br.src_x);
    let src_y = (1.0 - v) * ((1.0 - u) * tl.src_y + u * tr.src_y)
              + v * ((1.0 - u) * bl.src_y + u * br.src_y);

    (src_x, src_y)
}

#[rasmcore_macros::register_filter(
    name = "mesh_warp",
    category = "distortion",
    reference = "grid mesh warp distortion"
)]
pub fn mesh_warp(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &MeshWarpParams,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    let cols = config.grid_cols;
    let rows = config.grid_rows;
    if cols < 2 || rows < 2 {
        return Err(ImageError::InvalidParameters(
            "mesh_warp: grid must be at least 2x2".into(),
        ));
    }

    if is_16bit(info.format) {
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = upstream(full)?;
        let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
        return process_via_8bit(&pixels, info16, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            mesh_warp(r, &mut u, i8, config)
        });
    }

    let w = info.width as f32;
    let h = info.height as f32;
    let expected = (cols * rows) as usize;

    // Parse grid or use identity if empty
    let grid = if config.grid_json.trim().is_empty() {
        identity_grid(cols, rows, w, h)
    } else {
        parse_grid(&config.grid_json, w, h, expected)?
    };

    let dummy_j = crate::domain::ewa::JACOBIAN_IDENTITY;

    apply_distortion(
        request,
        upstream,
        info,
        DistortionOverlap::FullImage,
        DistortionSampling::Bilinear,
        &|xf, yf| {
            if let Some((c, r)) = find_quad(&grid, cols, xf, yf, rows) {
                let tl = &grid[(r * cols + c) as usize];
                let tr = &grid[(r * cols + c + 1) as usize];
                let bl = &grid[((r + 1) * cols + c) as usize];
                let br = &grid[((r + 1) * cols + c + 1) as usize];
                bilinear_inverse(xf, yf, tl, tr, bl, br)
            } else {
                // Outside grid — identity
                (xf, yf)
            }
        },
        &|_xf, _yf| dummy_j,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    fn make_rgba(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                pixels[i] = x as u8;
                pixels[i + 1] = y as u8;
                pixels[i + 2] = 128;
                pixels[i + 3] = 255;
            }
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn identity_grid_produces_unchanged_output() {
        let (pixels, info) = make_rgba(16, 16);
        let config = MeshWarpParams {
            grid_cols: 3,
            grid_rows: 3,
            grid_json: String::new(), // empty = identity
        };
        let request = Rect::new(0, 0, 16, 16);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = mesh_warp(request, &mut u, &info, &config).unwrap();
        assert_eq!(result.len(), pixels.len());
        // Identity warp should produce pixels very close to input
        // (bilinear sampling may cause sub-pixel differences at boundaries)
        let mut max_diff = 0u8;
        for i in 0..pixels.len() {
            let diff = (result[i] as i16 - pixels[i] as i16).unsigned_abs() as u8;
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(max_diff <= 1, "identity warp max diff was {max_diff}, expected <= 1");
    }

    #[test]
    fn uniform_translation_shifts_image() {
        let (pixels, info) = make_rgba(32, 32);
        // Shift the destination grid 2 pixels right (dst = src + 2px in x)
        // This means each output pixel maps to a source pixel 2px to its left
        let mut grid = String::from("[");
        let cols = 3u32;
        let rows = 3u32;
        for r in 0..rows {
            for c in 0..cols {
                let sx = c as f32 / (cols - 1) as f32;
                let sy = r as f32 / (rows - 1) as f32;
                let dx = sx + 2.0 / 32.0; // shift dst 2px right
                let dy = sy;
                if r > 0 || c > 0 {
                    grid.push_str(", ");
                }
                grid.push_str(&format!("[{sx}, {sy}, {dx}, {dy}]"));
            }
        }
        grid.push(']');

        let config = MeshWarpParams {
            grid_cols: cols,
            grid_rows: rows,
            grid_json: grid,
        };
        let request = Rect::new(0, 0, 32, 32);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = mesh_warp(request, &mut u, &info, &config).unwrap();

        // Pixels at x=4..28 should have R channel close to (x - 2)
        // (shifted left by 2 in source)
        let mut close_count = 0;
        let total = 20 * 28; // x=4..24, y=2..30
        for y in 2..30 {
            for x in 4..24 {
                let i = ((y * 32 + x) * 4) as usize;
                let expected_r = (x - 2) as u8;
                let diff = (result[i] as i16 - expected_r as i16).unsigned_abs();
                if diff <= 2 {
                    close_count += 1;
                }
            }
        }
        let pct = close_count as f32 / total as f32;
        assert!(
            pct > 0.8,
            "expected >80% of interior pixels to be shifted, got {:.1}%",
            pct * 100.0
        );
    }
}
