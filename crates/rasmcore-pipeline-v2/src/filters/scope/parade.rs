//! Parade scope implementation.

use super::{new_scope_buf, clamp_buf, plot_dot, draw_line};

/// Parade scope — R/G/B side-by-side waveforms.
pub fn compute_parade(input: &[f32], w: u32, h: u32, size: u32, _log_scale: bool) -> Vec<f32> {
    let mut buf = new_scope_buf(size);
    let third = size / 3;
    let intensity = (4.0 / h as f32).max(0.02).min(0.5);

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize * 4;
            let r = input[idx].clamp(0.0, 1.0);
            let g = input[idx + 1].clamp(0.0, 1.0);
            let b = input[idx + 2].clamp(0.0, 1.0);

            let sx = (x as f32 / w as f32 * (third - 2) as f32) as i32;

            // Red channel (left third)
            let ry = ((1.0 - r) * (size - 1) as f32) as i32;
            plot_dot(&mut buf, size, sx, ry, 0.9, 0.15, 0.15, intensity);

            // Green channel (middle third)
            let gy = ((1.0 - g) * (size - 1) as f32) as i32;
            plot_dot(&mut buf, size, sx + third as i32, gy, 0.15, 0.9, 0.15, intensity);

            // Blue channel (right third)
            let by = ((1.0 - b) * (size - 1) as f32) as i32;
            plot_dot(&mut buf, size, sx + 2 * third as i32, by, 0.15, 0.15, 0.9, intensity);
        }
    }

    // Separator lines
    draw_line(&mut buf, size, third as i32, 0, third as i32, size as i32 - 1, 0.3, 0.3, 0.3, 1.0);
    draw_line(&mut buf, size, 2 * third as i32, 0, 2 * third as i32, size as i32 - 1, 0.3, 0.3, 0.3, 1.0);

    clamp_buf(&mut buf);
    buf
}
