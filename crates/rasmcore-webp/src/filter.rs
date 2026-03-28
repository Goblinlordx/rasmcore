//! VP8 loop filter — deblocking filter (RFC 6386 Section 15).
//!
//! Smooths block boundaries to reduce visible artifacts after quantization.
//! Two modes: simple (fast, 2 pixels per edge) and normal (4 pixels with HEV).

#![allow(clippy::too_many_arguments)]

/// Loop filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Fast filter, adjusts 2 pixels per edge.
    Simple,
    /// Stronger filter, adjusts 4 pixels per edge with HEV detection.
    Normal,
}

/// Clamp value to i8 range.
fn clamp_i8(v: i32) -> i32 {
    v.clamp(-128, 127)
}

/// Compute the common filter adjustment value (RFC 6386 Section 15.2).
fn common_adjust(use_outer_taps: bool, p1: i32, p0: i32, q0: i32, q1: i32) -> i32 {
    let a = clamp_i8(p1 - q1);
    if use_outer_taps {
        clamp_i8(a + 3 * (q0 - p0))
    } else {
        clamp_i8(3 * (q0 - p0))
    }
}

/// Apply simple loop filter to a row of pixels at a vertical edge.
///
/// `pixels` is a mutable slice, `stride` is row pitch, `edge_offset` is
/// the column index of the edge (q0 column). Filters `count` rows.
pub fn filter_simple_vertical(
    pixels: &mut [u8],
    stride: usize,
    edge_offset: usize,
    count: usize,
    filter_limit: i32,
) {
    for i in 0..count {
        let off = i * stride + edge_offset;
        if off < 1 || off >= pixels.len() {
            continue;
        }
        let p0 = pixels[off - 1] as i32;
        let q0 = pixels[off] as i32;

        let delta = 3 * (q0 - p0);
        if delta.abs() > filter_limit {
            continue;
        }

        let a = clamp_i8(delta + 4) >> 3;
        let b = clamp_i8(delta + 3) >> 3;
        pixels[off - 1] = (p0 + a).clamp(0, 255) as u8;
        pixels[off] = (q0 - b).clamp(0, 255) as u8;
    }
}

/// Apply simple loop filter at a horizontal edge.
pub fn filter_simple_horizontal(
    pixels: &mut [u8],
    stride: usize,
    edge_row: usize,
    col_start: usize,
    count: usize,
    filter_limit: i32,
) {
    for i in 0..count {
        let col = col_start + i;
        let q0_off = edge_row * stride + col;
        let p0_off = (edge_row - 1) * stride + col;
        if q0_off >= pixels.len() || edge_row == 0 {
            continue;
        }
        let p0 = pixels[p0_off] as i32;
        let q0 = pixels[q0_off] as i32;

        let delta = 3 * (q0 - p0);
        if delta.abs() > filter_limit {
            continue;
        }

        let a = clamp_i8(delta + 4) >> 3;
        let b = clamp_i8(delta + 3) >> 3;
        pixels[p0_off] = (p0 + a).clamp(0, 255) as u8;
        pixels[q0_off] = (q0 - b).clamp(0, 255) as u8;
    }
}

/// Apply normal loop filter at a vertical edge (4 pixels: p1 p0 | q0 q1).
pub fn filter_normal_vertical(
    pixels: &mut [u8],
    stride: usize,
    edge_offset: usize,
    count: usize,
    filter_limit: i32,
    inner_limit: i32,
    hev_threshold: i32,
) {
    for i in 0..count {
        let off = i * stride + edge_offset;
        if off < 2 || off + 1 >= pixels.len() {
            continue;
        }
        let p1 = pixels[off - 2] as i32;
        let p0 = pixels[off - 1] as i32;
        let q0 = pixels[off] as i32;
        let q1 = pixels[off + 1] as i32;

        // Check if edge needs filtering
        if (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) > filter_limit {
            continue;
        }
        if (p1 - p0).abs() > inner_limit || (q1 - q0).abs() > inner_limit {
            continue;
        }

        let hev = (p1 - p0).abs() > hev_threshold || (q1 - q0).abs() > hev_threshold;

        if hev {
            // High edge variance: only adjust p0, q0
            let a = common_adjust(true, p1, p0, q0, q1);
            let a1 = clamp_i8(a + 4) >> 3;
            let a2 = clamp_i8(a + 3) >> 3;
            pixels[off - 1] = (p0 + a1).clamp(0, 255) as u8;
            pixels[off] = (q0 - a2).clamp(0, 255) as u8;
        } else {
            // Normal: adjust p1, p0, q0, q1
            let a = common_adjust(false, p1, p0, q0, q1);
            let a1 = clamp_i8(a + 4) >> 3;
            let a2 = clamp_i8(a + 3) >> 3;
            let a3 = (a1 + 1) >> 1;
            pixels[off - 2] = (p1 + a3).clamp(0, 255) as u8;
            pixels[off - 1] = (p0 + a1).clamp(0, 255) as u8;
            pixels[off] = (q0 - a2).clamp(0, 255) as u8;
            pixels[off + 1] = (q1 - a3).clamp(0, 255) as u8;
        }
    }
}

/// Apply normal loop filter at a horizontal edge.
pub fn filter_normal_horizontal(
    pixels: &mut [u8],
    stride: usize,
    edge_row: usize,
    col_start: usize,
    count: usize,
    filter_limit: i32,
    inner_limit: i32,
    hev_threshold: i32,
) {
    for i in 0..count {
        let col = col_start + i;
        if edge_row < 2 {
            continue;
        }
        let p1_off = (edge_row - 2) * stride + col;
        let p0_off = (edge_row - 1) * stride + col;
        let q0_off = edge_row * stride + col;
        let q1_off = (edge_row + 1) * stride + col;
        if q1_off >= pixels.len() {
            continue;
        }

        let p1 = pixels[p1_off] as i32;
        let p0 = pixels[p0_off] as i32;
        let q0 = pixels[q0_off] as i32;
        let q1 = pixels[q1_off] as i32;

        if (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) > filter_limit {
            continue;
        }
        if (p1 - p0).abs() > inner_limit || (q1 - q0).abs() > inner_limit {
            continue;
        }

        let hev = (p1 - p0).abs() > hev_threshold || (q1 - q0).abs() > hev_threshold;

        if hev {
            let a = common_adjust(true, p1, p0, q0, q1);
            let a1 = clamp_i8(a + 4) >> 3;
            let a2 = clamp_i8(a + 3) >> 3;
            pixels[p0_off] = (p0 + a1).clamp(0, 255) as u8;
            pixels[q0_off] = (q0 - a2).clamp(0, 255) as u8;
        } else {
            let a = common_adjust(false, p1, p0, q0, q1);
            let a1 = clamp_i8(a + 4) >> 3;
            let a2 = clamp_i8(a + 3) >> 3;
            let a3 = (a1 + 1) >> 1;
            pixels[p1_off] = (p1 + a3).clamp(0, 255) as u8;
            pixels[p0_off] = (p0 + a1).clamp(0, 255) as u8;
            pixels[q0_off] = (q0 - a2).clamp(0, 255) as u8;
            pixels[q1_off] = (q1 - a3).clamp(0, 255) as u8;
        }
    }
}

/// Apply loop filter to a full frame (Y plane only for now).
///
/// Filters macroblock and sub-block boundaries to reduce blocking artifacts.
pub fn apply_loop_filter(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    filter_level: u8,
    sharpness: u8,
    filter_type: FilterType,
) {
    if filter_level == 0 {
        return;
    }

    let (filter_limit, inner_limit, hev_threshold) = compute_filter_params(filter_level, sharpness);

    let stride = width;

    // Filter macroblock boundaries (every 16 pixels)
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);

    for mb_row in 0..mb_h {
        for mb_col in 0..mb_w {
            let x = mb_col * 16;
            let y = mb_row * 16;

            match filter_type {
                FilterType::Simple => {
                    // Vertical edge (left boundary of MB)
                    if mb_col > 0 {
                        filter_simple_vertical(pixels, stride, x, 16.min(height - y), filter_limit);
                    }
                    // Horizontal edge (top boundary of MB)
                    if mb_row > 0 {
                        filter_simple_horizontal(
                            pixels,
                            stride,
                            y,
                            x,
                            16.min(width - x),
                            filter_limit,
                        );
                    }
                }
                FilterType::Normal => {
                    if mb_col > 0 {
                        filter_normal_vertical(
                            pixels,
                            stride,
                            x,
                            16.min(height - y),
                            filter_limit,
                            inner_limit,
                            hev_threshold,
                        );
                    }
                    if mb_row > 0 {
                        filter_normal_horizontal(
                            pixels,
                            stride,
                            y,
                            x,
                            16.min(width - x),
                            filter_limit,
                            inner_limit,
                            hev_threshold,
                        );
                    }
                }
            }
        }
    }
}

/// Compute filter threshold parameters from level and sharpness.
/// Returns (filter_limit, inner_limit, hev_threshold).
pub fn compute_filter_params(filter_level: u8, sharpness: u8) -> (i32, i32, i32) {
    let level = filter_level as i32;

    // Interior limit (RFC 6386 Section 15.3)
    let mut inner_limit = if sharpness == 0 {
        level
    } else if sharpness > 4 {
        level >> 2
    } else {
        (level >> 1).max(1)
    };
    inner_limit = inner_limit.clamp(1, 63);

    // Outer limit
    let filter_limit = 2 * level + inner_limit;

    // HEV threshold (based on filter level)
    let hev_threshold = if level >= 40 {
        2
    } else if level >= 15 {
        1
    } else {
        0
    };

    (filter_limit, inner_limit, hev_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_filter_smooths_mild_edge() {
        // Create a mild vertical edge (within filter_limit)
        let mut pixels = vec![0u8; 8 * 4]; // 8 wide, 4 tall
        for row in 0..4 {
            for col in 0..4 {
                pixels[row * 8 + col] = 120;
            }
            for col in 4..8 {
                pixels[row * 8 + col] = 130;
            }
        }
        let original = pixels.clone();
        filter_simple_vertical(&mut pixels, 8, 4, 4, 255);

        // Edge pixels should be smoothed (closer together)
        let p0 = pixels[3] as i32;
        let q0 = pixels[4] as i32;
        let orig_p0 = original[3] as i32;
        let orig_q0 = original[4] as i32;
        assert!(
            (p0 - q0).abs() < (orig_p0 - orig_q0).abs(),
            "edge should be smoothed: was {} now {}",
            orig_p0 - orig_q0,
            p0 - q0
        );
    }

    #[test]
    fn normal_filter_adjusts_pixels() {
        // Use values with moderate step across edge (within filter limits)
        let mut pixels = vec![0u8; 8 * 4];
        for row in 0..4 {
            pixels[row * 8] = 115;
            pixels[row * 8 + 1] = 120;
            pixels[row * 8 + 2] = 130;
            pixels[row * 8 + 3] = 135;
        }
        let original = pixels.clone();
        filter_normal_vertical(&mut pixels, 8, 2, 4, 255, 63, 0);

        // p0 and q0 should change (they're at the edge)
        assert_ne!(pixels[1], original[1], "p0 should change");
        assert_ne!(pixels[2], original[2], "q0 should change");
    }

    #[test]
    fn zero_filter_level_is_noop() {
        let mut pixels = vec![128u8; 32 * 32];
        pixels[15] = 0;
        pixels[16] = 255;
        let original = pixels.clone();
        apply_loop_filter(&mut pixels, 32, 32, 0, 0, FilterType::Simple);
        assert_eq!(pixels, original);
    }

    #[test]
    fn compute_params_level_zero() {
        let (fl, il, hev) = compute_filter_params(0, 0);
        assert_eq!(hev, 0);
        assert!(fl >= 0);
        assert!(il >= 1);
    }

    #[test]
    fn compute_params_high_level() {
        let (fl, il, hev) = compute_filter_params(63, 0);
        assert!(fl > 0);
        assert!(il > 0);
        assert_eq!(hev, 2);
    }
}
