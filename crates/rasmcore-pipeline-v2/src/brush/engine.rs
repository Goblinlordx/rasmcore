//! CPU brush engine — orchestrates path processing, dab generation, and compositing.

use super::composite::{composite_dab, composite_stroke_onto_layer};
use super::dab::evaluate_dynamics;
use super::path::{smooth_catmull_rom, spacing_interpolation};
use super::types::{AccumulationBuffer, BrushParams, StrokePoint};

/// CPU brush engine — renders a stroke onto an f32 RGBA layer.
pub struct CpuBrushEngine;

impl CpuBrushEngine {
    /// Render a complete stroke onto a layer.
    ///
    /// - `layer`: f32 RGBA pixel data (width × height × 4). Modified in place.
    /// - `width`, `height`: layer dimensions.
    /// - `points`: raw stroke input points.
    /// - `params`: brush parameters.
    /// - `color`: stroke color [R, G, B, A].
    pub fn render_stroke(
        layer: &mut [f32],
        width: u32,
        height: u32,
        points: &[StrokePoint],
        params: &BrushParams,
        color: [f32; 4],
    ) {
        if points.is_empty() || params.diameter < 0.5 {
            return;
        }

        // 1. Smooth the path
        let subdivisions = if params.smoothing > 0.01 { 4 } else { 1 };
        let smoothed = smooth_catmull_rom(points, params.smoothing, subdivisions);

        // 2. Spacing interpolation — evenly-spaced dab positions
        let spacing_px = (params.diameter * params.spacing).max(0.5);
        let dab_positions = spacing_interpolation(&smoothed, spacing_px);

        if dab_positions.is_empty() {
            return;
        }

        // 3. Accumulation buffer for the stroke
        let mut accum = AccumulationBuffer::new(width, height);

        // 4. Generate and composite each dab
        for (i, pos) in dab_positions.iter().enumerate() {
            let dab = evaluate_dynamics(pos, params, i as u32);
            composite_dab(&mut accum, &dab, color);
        }

        // 5. Composite stroke onto layer with stroke opacity
        composite_stroke_onto_layer(layer, &accum, params.opacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(x: f32, y: f32, pressure: f32) -> StrokePoint {
        StrokePoint {
            x,
            y,
            pressure,
            tilt_x: 0.0,
            tilt_y: 0.0,
            rotation: 0.0,
            velocity: 0.5,
            timestamp: 0.0,
        }
    }

    #[test]
    fn render_stroke_produces_visible_output() {
        let w = 64u32;
        let h = 64u32;
        let mut layer = vec![0.0f32; (w * h * 4) as usize];

        let points = vec![make_point(10.0, 32.0, 1.0), make_point(54.0, 32.0, 1.0)];
        let params = BrushParams {
            diameter: 8.0,
            spacing: 0.25,
            hardness: 1.0,
            flow: 1.0,
            opacity: 1.0,
            ..BrushParams::default()
        };

        CpuBrushEngine::render_stroke(&mut layer, w, h, &points, &params, [1.0, 0.0, 0.0, 1.0]);

        // Should have red pixels along y=32
        let mid_idx = (32 * w + 30) as usize * 4;
        assert!(
            layer[mid_idx] > 0.5,
            "red channel at midpoint: {}",
            layer[mid_idx]
        );
        assert!(
            layer[mid_idx + 3] > 0.5,
            "alpha at midpoint: {}",
            layer[mid_idx + 3]
        );

        // Corners should be untouched
        assert!(layer[0] < 0.01, "corner should be blank");
    }

    #[test]
    fn render_stroke_correct_dab_count() {
        let w = 128u32;
        let h = 128u32;
        let mut layer = vec![0.0f32; (w * h * 4) as usize];

        // Straight line of 100px with 10px spacing = ~10 dabs
        let points = vec![make_point(10.0, 64.0, 1.0), make_point(110.0, 64.0, 1.0)];
        let params = BrushParams {
            diameter: 10.0,
            spacing: 1.0, // spacing = 1.0 × diameter = 10px
            hardness: 1.0,
            flow: 1.0,
            opacity: 1.0,
            smoothing: 0.0,
            ..BrushParams::default()
        };

        CpuBrushEngine::render_stroke(&mut layer, w, h, &points, &params, [1.0, 1.0, 1.0, 1.0]);

        // Count pixels with alpha > 0.5 along the stroke line
        let mut marked_pixels = 0;
        for x in 0..w {
            let idx = (64 * w + x) as usize * 4;
            if layer[idx + 3] > 0.5 {
                marked_pixels += 1;
            }
        }
        assert!(
            marked_pixels > 30,
            "should mark substantial portion of stroke path, got {marked_pixels}"
        );
    }

    #[test]
    fn render_stroke_pressure_varies_size() {
        let w = 128u32;
        let h = 128u32;
        let mut layer = vec![0.0f32; (w * h * 4) as usize];

        // Stroke with varying pressure
        let points = vec![make_point(20.0, 64.0, 0.1), make_point(108.0, 64.0, 1.0)];
        let params = BrushParams {
            diameter: 20.0,
            spacing: 0.25,
            hardness: 1.0,
            flow: 1.0,
            opacity: 1.0,
            dynamics: super::super::types::DynamicsCurves {
                pressure_size: super::super::types::DynamicsCurve {
                    points: vec![(0.0, 0.1), (1.0, 1.0)],
                },
                ..super::super::types::DynamicsCurves::default()
            },
            ..BrushParams::default()
        };

        CpuBrushEngine::render_stroke(&mut layer, w, h, &points, &params, [1.0, 1.0, 1.0, 1.0]);

        // Count marked pixels in the first and last quarters
        let count_in_range = |x_start: u32, x_end: u32| -> u32 {
            let mut count = 0;
            for y in 50..80 {
                for x in x_start..x_end {
                    let idx = (y * w + x) as usize * 4;
                    if layer[idx + 3] > 0.3 {
                        count += 1;
                    }
                }
            }
            count
        };

        let low_pressure_area = count_in_range(10, 40);
        let high_pressure_area = count_in_range(80, 120);
        assert!(
            high_pressure_area > low_pressure_area,
            "high pressure ({high_pressure_area}) should cover more than low ({low_pressure_area})"
        );
    }
}
