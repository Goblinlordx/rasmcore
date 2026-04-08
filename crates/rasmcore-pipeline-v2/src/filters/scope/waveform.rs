//! Waveform scope implementation.

use super::{new_scope_buf, clamp_buf, plot_dot};

/// Waveform scope — per-column brightness distribution.
pub fn compute_waveform(input: &[f32], w: u32, h: u32, size: u32, _log_scale: bool) -> Vec<f32> {
    let mut buf = new_scope_buf(size);
    let intensity = (4.0 / h as f32).max(0.02).min(0.5); // brighter dots for fewer rows

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize * 4;
            let r = input[idx];
            let g = input[idx + 1];
            let b = input[idx + 2];
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            let sx = (x as f32 / w as f32 * (size - 1) as f32) as i32;
            let sy = ((1.0 - luma.clamp(0.0, 1.0)) * (size - 1) as f32) as i32;

            plot_dot(&mut buf, size, sx, sy, 0.2, 0.8, 0.2, intensity);
        }
    }

    clamp_buf(&mut buf);
    buf
}
