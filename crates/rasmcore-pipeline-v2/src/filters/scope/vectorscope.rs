//! Vectorscope scope implementation.

use super::super::helpers::rgb_to_hsl;
use super::{clamp_buf, draw_line, new_scope_buf, plot_dot};

/// Vectorscope — chrominance polar plot.
pub fn compute_vectorscope(
    input: &[f32],
    _w: u32,
    _h: u32,
    size: u32,
    _log_scale: bool,
) -> Vec<f32> {
    let mut buf = new_scope_buf(size);
    let center = size as f32 / 2.0;
    let radius = center - 2.0;
    let pixel_count = input.len() / 4;
    let intensity = (8.0 / pixel_count as f32).max(0.005).min(0.3);

    // Draw graticule: circle outline and color targets
    let steps = (size as f32 * std::f32::consts::PI) as i32;
    for i in 0..steps {
        let angle = i as f32 / steps as f32 * std::f32::consts::TAU;
        let x = (center + radius * angle.cos()) as i32;
        let y = (center - radius * angle.sin()) as i32;
        plot_dot(&mut buf, size, x, y, 0.2, 0.2, 0.2, 1.0);
    }
    // Crosshair
    draw_line(
        &mut buf,
        size,
        center as i32,
        0,
        center as i32,
        size as i32 - 1,
        0.15,
        0.15,
        0.15,
        1.0,
    );
    draw_line(
        &mut buf,
        size,
        0,
        center as i32,
        size as i32 - 1,
        center as i32,
        0.15,
        0.15,
        0.15,
        1.0,
    );

    // Skin tone line (~123° from positive I axis, roughly 33° from B-Y axis)
    let skin_angle = 123.0f32.to_radians();
    let sx = (center + radius * skin_angle.cos()) as i32;
    let sy = (center - radius * skin_angle.sin()) as i32;
    draw_line(
        &mut buf,
        size,
        center as i32,
        center as i32,
        sx,
        sy,
        0.5,
        0.4,
        0.3,
        0.6,
    );

    // Color target markers (R, G, B, Cy, Mg, Yl at standard positions)
    let targets: [(f32, f32, f32, f32); 6] = [
        (0.0, 1.0, 0.0, 0.0),   // Red at 0°
        (120.0, 0.0, 1.0, 0.0), // Green at 120°
        (240.0, 0.0, 0.0, 1.0), // Blue at 240°
        (180.0, 0.0, 0.8, 0.8), // Cyan at 180°
        (300.0, 0.8, 0.0, 0.8), // Magenta at 300°
        (60.0, 0.8, 0.8, 0.0),  // Yellow at 60°
    ];
    for (angle_deg, tr, tg, tb) in targets {
        let a = angle_deg.to_radians();
        let tx = (center + radius * 0.75 * a.cos()) as i32;
        let ty = (center - radius * 0.75 * a.sin()) as i32;
        for dx in -2..=2i32 {
            for dy in -2..=2i32 {
                plot_dot(&mut buf, size, tx + dx, ty + dy, tr, tg, tb, 0.7);
            }
        }
    }

    // Plot each pixel's chrominance
    for pixel in input.chunks_exact(4) {
        let r = pixel[0].clamp(0.0, 1.0);
        let g = pixel[1].clamp(0.0, 1.0);
        let b = pixel[2].clamp(0.0, 1.0);
        let (h, s, _l) = rgb_to_hsl(r, g, b);

        if s < 0.001 {
            continue;
        } // skip achromatic

        let angle = h.to_radians();
        let dist = s * radius;
        let sx = (center + dist * angle.cos()) as i32;
        let sy = (center - dist * angle.sin()) as i32;

        // Dot color matches the pixel's color (dimmed)
        plot_dot(
            &mut buf,
            size,
            sx,
            sy,
            r * 0.6 + 0.2,
            g * 0.6 + 0.2,
            b * 0.6 + 0.2,
            intensity,
        );
    }

    clamp_buf(&mut buf);
    buf
}
