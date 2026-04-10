//! Dab stamp compositing onto the accumulation buffer.

use super::dab::generate_stamp;
use super::types::{AccumulationBuffer, DabInstance};

/// Composite a dab stamp onto the accumulation buffer.
///
/// Uses max-opacity blending: each pixel retains the maximum alpha
/// contribution from any dab, preventing darkening from overlap.
/// This matches Photoshop's stroke accumulation model.
pub fn composite_dab(buf: &mut AccumulationBuffer, dab: &DabInstance, color: [f32; 4]) {
    let stamp_size = dab.size.round().max(1.0) as u32;
    if stamp_size == 0 {
        return;
    }

    let stamp = generate_stamp(stamp_size, dab.hardness, dab.roundness, dab.angle);
    let half = stamp_size as f32 * 0.5;
    let start_x = (dab.x - half).floor() as i32;
    let start_y = (dab.y - half).floor() as i32;

    for sy in 0..stamp_size {
        let py = start_y + sy as i32;
        if py < 0 || py >= buf.height as i32 {
            continue;
        }
        for sx in 0..stamp_size {
            let px = start_x + sx as i32;
            if px < 0 || px >= buf.width as i32 {
                continue;
            }

            let stamp_alpha = stamp[(sy * stamp_size + sx) as usize] * dab.opacity;
            if stamp_alpha < 1e-6 {
                continue;
            }

            let idx = ((py as u32 * buf.width + px as u32) * 4) as usize;
            let existing_alpha = buf.data[idx + 3];

            // Max-opacity blend: use the higher alpha, blend color accordingly
            if stamp_alpha > existing_alpha {
                let sa = stamp_alpha;
                buf.data[idx] = color[0] * sa;
                buf.data[idx + 1] = color[1] * sa;
                buf.data[idx + 2] = color[2] * sa;
                buf.data[idx + 3] = sa;
            }
        }
    }
}

/// Composite the stroke accumulation buffer onto the source layer.
///
/// Uses standard "over" compositing with stroke opacity.
pub fn composite_stroke_onto_layer(
    layer: &mut [f32],
    accum: &AccumulationBuffer,
    stroke_opacity: f32,
) {
    for i in 0..(accum.width * accum.height) as usize {
        let idx = i * 4;
        let sa = accum.data[idx + 3] * stroke_opacity;
        if sa < 1e-6 {
            continue;
        }
        // Premultiplied over composite
        let inv_sa = 1.0 - sa;
        layer[idx] = accum.data[idx] * stroke_opacity + layer[idx] * inv_sa;
        layer[idx + 1] = accum.data[idx + 1] * stroke_opacity + layer[idx + 1] * inv_sa;
        layer[idx + 2] = accum.data[idx + 2] * stroke_opacity + layer[idx + 2] * inv_sa;
        layer[idx + 3] = sa + layer[idx + 3] * inv_sa;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_dab_leaves_mark() {
        let mut buf = AccumulationBuffer::new(32, 32);
        let dab = DabInstance {
            x: 16.0,
            y: 16.0,
            size: 10.0,
            opacity: 1.0,
            angle: 0.0,
            roundness: 1.0,
            hardness: 1.0,
        };
        composite_dab(&mut buf, &dab, [1.0, 0.0, 0.0, 1.0]);
        // Center should have color
        let idx = (16 * 32 + 16) * 4;
        assert!(buf.data[idx as usize] > 0.5, "red channel should be set");
        assert!(buf.data[idx as usize + 3] > 0.5, "alpha should be set");
    }

    #[test]
    fn composite_dab_respects_bounds() {
        let mut buf = AccumulationBuffer::new(16, 16);
        // Dab near edge — should not panic
        let dab = DabInstance {
            x: 1.0,
            y: 1.0,
            size: 10.0,
            opacity: 1.0,
            angle: 0.0,
            roundness: 1.0,
            hardness: 1.0,
        };
        composite_dab(&mut buf, &dab, [1.0, 1.0, 1.0, 1.0]);
        // Should not crash — dab extends past (0,0)
    }

    #[test]
    fn max_opacity_blend_no_darkening() {
        let mut buf = AccumulationBuffer::new(4, 4);
        let dab = DabInstance {
            x: 2.0,
            y: 2.0,
            size: 4.0,
            opacity: 0.5,
            angle: 0.0,
            roundness: 1.0,
            hardness: 1.0,
        };
        composite_dab(&mut buf, &dab, [1.0, 0.0, 0.0, 1.0]);
        let alpha_1 = buf.data[(2 * 4 + 2) * 4 + 3];

        // Second dab at same position — alpha should not increase beyond first
        composite_dab(&mut buf, &dab, [1.0, 0.0, 0.0, 1.0]);
        let alpha_2 = buf.data[(2 * 4 + 2) * 4 + 3];
        assert!(
            (alpha_1 - alpha_2).abs() < 1e-6,
            "max-opacity should not accumulate"
        );
    }

    #[test]
    fn stroke_composite_onto_layer() {
        let mut layer = vec![
            0.5f32, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0,
        ];
        let mut accum = AccumulationBuffer::new(2, 2);
        // Put a red mark at (0,0)
        accum.data[0] = 1.0; // R premultiplied
        accum.data[3] = 1.0; // A
        composite_stroke_onto_layer(&mut layer, &accum, 1.0);
        // Pixel (0,0) should now be red (stroke over gray)
        assert!(layer[0] > 0.9, "should be mostly red");
        // Pixel (1,0) should be unchanged
        assert!((layer[4] - 0.5).abs() < 1e-6);
    }
}
