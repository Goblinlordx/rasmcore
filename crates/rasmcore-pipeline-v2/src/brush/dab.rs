//! Dab generation — evaluate dynamics and generate stamp instances.

use super::types::{BrushParams, DabInstance, StrokePoint};

/// Evaluate dynamics curves and produce a DabInstance from a stroke point.
pub fn evaluate_dynamics(point: &StrokePoint, params: &BrushParams, rng_seed: u32) -> DabInstance {
    let dynamics = &params.dynamics;

    // Map pressure/velocity to size/opacity via curves
    let size_mult = dynamics.pressure_size.evaluate(point.pressure)
        * dynamics.velocity_size.evaluate(point.velocity.clamp(0.0, 1.0));
    let opacity_mult = dynamics.pressure_opacity.evaluate(point.pressure);
    let angle_offset = dynamics.tilt_angle.evaluate(
        (point.tilt_x * point.tilt_x + point.tilt_y * point.tilt_y).sqrt().clamp(0.0, 1.0),
    ) * std::f32::consts::PI;

    // Apply scatter — random perpendicular displacement
    let (scatter_x, scatter_y) = if params.scatter > 0.0 {
        let hash = simple_hash(rng_seed);
        let angle = hash * std::f32::consts::TAU;
        let dist = params.scatter * params.diameter * simple_hash(rng_seed.wrapping_add(1));
        (angle.cos() * dist, angle.sin() * dist)
    } else {
        (0.0, 0.0)
    };

    DabInstance {
        x: point.x + scatter_x,
        y: point.y + scatter_y,
        size: params.diameter * size_mult,
        opacity: params.flow * opacity_mult,
        angle: params.angle + angle_offset,
        roundness: params.roundness,
        hardness: params.hardness,
    }
}

/// Generate the stamp alpha mask for a dab (parametric circle with hardness falloff).
///
/// Returns a square buffer of `size × size` alpha values [0, 1].
/// `hardness`: 0.0 = gaussian-like falloff, 1.0 = sharp circle edge.
pub fn generate_stamp(size: u32, hardness: f32, roundness: f32, angle: f32) -> Vec<f32> {
    if size == 0 {
        return vec![];
    }
    let r = size as f32 * 0.5;
    let center = r;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let inv_roundness = if roundness > 1e-6 { 1.0 / roundness } else { 100.0 };

    let mut stamp = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            // Rotate and apply roundness
            let rx = dx * cos_a + dy * sin_a;
            let ry = (-dx * sin_a + dy * cos_a) * inv_roundness;
            let dist = (rx * rx + ry * ry).sqrt() / r;

            let alpha = if dist >= 1.0 {
                0.0
            } else if hardness >= 1.0 {
                1.0
            } else {
                // Smooth falloff: ramp from hardness to 1.0
                let inner = hardness;
                if dist <= inner {
                    1.0
                } else {
                    let t = (dist - inner) / (1.0 - inner);
                    (1.0 - t * t).max(0.0)
                }
            };
            stamp.push(alpha);
        }
    }
    stamp
}

/// Simple deterministic hash for scatter (not cryptographic).
fn simple_hash(seed: u32) -> f32 {
    let mut h = seed;
    h = h.wrapping_mul(0x9e3779b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    (h & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::types::{BrushParams, DynamicsCurve, DynamicsCurves};

    #[test]
    fn stamp_circle_center_opaque() {
        let stamp = generate_stamp(16, 0.8, 1.0, 0.0);
        assert_eq!(stamp.len(), 16 * 16);
        // Center pixel should be fully opaque
        assert!(stamp[8 * 16 + 8] > 0.99);
    }

    #[test]
    fn stamp_circle_corners_transparent() {
        let stamp = generate_stamp(16, 1.0, 1.0, 0.0);
        // Corner (0,0) is outside the circle
        assert!(stamp[0] < 0.01);
    }

    #[test]
    fn stamp_soft_edge_falloff() {
        let stamp = generate_stamp(32, 0.0, 1.0, 0.0);
        let center = stamp[16 * 32 + 16];
        let edge = stamp[16 * 32 + 30]; // near edge
        assert!(center > edge, "center should be brighter than edge");
    }

    #[test]
    fn dynamics_pressure_affects_size() {
        let point = StrokePoint {
            x: 50.0, y: 50.0, pressure: 0.5, tilt_x: 0.0, tilt_y: 0.0,
            rotation: 0.0, velocity: 0.0, timestamp: 0.0,
        };
        let params = BrushParams {
            diameter: 20.0,
            dynamics: DynamicsCurves {
                pressure_size: DynamicsCurve { points: vec![(0.0, 0.2), (1.0, 1.0)] },
                ..DynamicsCurves::default()
            },
            ..BrushParams::default()
        };
        let dab = evaluate_dynamics(&point, &params, 0);
        // Pressure 0.5 → size curve evaluates to 0.6 → diameter * 0.6 = 12
        assert!((dab.size - 12.0).abs() < 0.5, "got size {}", dab.size);
    }

    #[test]
    fn scatter_displaces_position() {
        let point = StrokePoint {
            x: 50.0, y: 50.0, pressure: 1.0, tilt_x: 0.0, tilt_y: 0.0,
            rotation: 0.0, velocity: 0.0, timestamp: 0.0,
        };
        let params = BrushParams {
            scatter: 1.0,
            diameter: 20.0,
            ..BrushParams::default()
        };
        let dab = evaluate_dynamics(&point, &params, 42);
        // With scatter=1.0 and diameter=20, displacement up to 20px
        let dist = ((dab.x - 50.0).powi(2) + (dab.y - 50.0).powi(2)).sqrt();
        assert!(dist > 0.0, "scatter should displace from center");
        assert!(dist <= 20.0, "displacement should not exceed diameter");
    }
}
