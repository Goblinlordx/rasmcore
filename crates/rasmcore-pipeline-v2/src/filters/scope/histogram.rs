//! Histogram scope implementation.

use super::{clamp_buf, fill_bar, new_scope_buf, plot_dot};

// ─── Scope implementations ─────────────────────────────────────────────────

/// Histogram scope — 256-bin per-channel distribution.
pub fn compute_histogram(input: &[f32], _w: u32, _h: u32, size: u32, log_scale: bool) -> Vec<f32> {
    let mut bins_r = [0u32; 256];
    let mut bins_g = [0u32; 256];
    let mut bins_b = [0u32; 256];
    let mut bins_l = [0u32; 256];

    for pixel in input.chunks_exact(4) {
        let r = (pixel[0].clamp(0.0, 1.0) * 255.0) as usize;
        let g = (pixel[1].clamp(0.0, 1.0) * 255.0) as usize;
        let b = (pixel[2].clamp(0.0, 1.0) * 255.0) as usize;
        let luma = (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]).clamp(0.0, 1.0);
        let l = (luma * 255.0) as usize;
        bins_r[r.min(255)] += 1;
        bins_g[g.min(255)] += 1;
        bins_b[b.min(255)] += 1;
        bins_l[l.min(255)] += 1;
    }

    // Find max for normalization
    let max_count = bins_r
        .iter()
        .chain(bins_g.iter())
        .chain(bins_b.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);

    let mut buf = new_scope_buf(size);

    for i in 0..256u32 {
        let x = (i as f32 / 255.0 * (size - 1) as f32) as u32;

        let normalize = |count: u32| -> u32 {
            let ratio = if log_scale {
                (count as f32 + 1.0).ln() / (max_count as f32 + 1.0).ln()
            } else {
                count as f32 / max_count as f32
            };
            (ratio * (size - 1) as f32) as u32
        };

        // Draw channels with partial transparency for overlap visibility
        fill_bar(
            &mut buf,
            size,
            x,
            normalize(bins_r[i as usize]),
            0.9,
            0.1,
            0.1,
            0.5,
        );
        fill_bar(
            &mut buf,
            size,
            x,
            normalize(bins_g[i as usize]),
            0.1,
            0.9,
            0.1,
            0.5,
        );
        fill_bar(
            &mut buf,
            size,
            x,
            normalize(bins_b[i as usize]),
            0.1,
            0.1,
            0.9,
            0.5,
        );
        // Luma outline (draw on top, thin)
        let lh = normalize(bins_l[i as usize]);
        if lh > 0 {
            let y = (size - 1 - lh) as i32;
            plot_dot(&mut buf, size, x as i32, y, 0.7, 0.7, 0.7, 0.6);
        }
    }

    clamp_buf(&mut buf);
    buf
}
