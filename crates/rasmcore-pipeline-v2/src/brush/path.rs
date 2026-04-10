//! Path processing — Catmull-Rom smoothing and spacing interpolation.

use super::types::StrokePoint;

/// Apply Catmull-Rom smoothing to a sequence of stroke points.
///
/// `strength` controls smoothing amount: 0.0 = no smoothing, 1.0 = full.
/// Returns a new path with interpolated points (original count × subdivision).
pub fn smooth_catmull_rom(
    points: &[StrokePoint],
    strength: f32,
    subdivisions: u32,
) -> Vec<StrokePoint> {
    if points.len() < 2 {
        return points.to_vec();
    }
    if strength < 1e-6 || subdivisions <= 1 {
        return points.to_vec();
    }

    let n = points.len();
    let mut result = Vec::with_capacity(n * subdivisions as usize);

    for i in 0..n - 1 {
        let p0 = if i > 0 { &points[i - 1] } else { &points[i] };
        let p1 = &points[i];
        let p2 = &points[i + 1];
        let p3 = if i + 2 < n {
            &points[i + 2]
        } else {
            &points[i + 1]
        };

        for s in 0..subdivisions {
            let t = s as f32 / subdivisions as f32;
            let raw = catmull_rom_interp(p0, p1, p2, p3, t);
            // Blend between original linear interp and smoothed
            let linear = StrokePoint::lerp(p1, p2, t);
            result.push(StrokePoint::lerp(&linear, &raw, strength));
        }
    }

    // Always include last point
    result.push(*points.last().unwrap());
    result
}

fn catmull_rom_interp(
    p0: &StrokePoint,
    p1: &StrokePoint,
    p2: &StrokePoint,
    p3: &StrokePoint,
    t: f32,
) -> StrokePoint {
    let cr = |a: f32, b: f32, c: f32, d: f32| -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * b)
            + (-a + c) * t
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
            + (-a + 3.0 * b - 3.0 * c + d) * t3)
    };
    StrokePoint {
        x: cr(p0.x, p1.x, p2.x, p3.x),
        y: cr(p0.y, p1.y, p2.y, p3.y),
        pressure: cr(p0.pressure, p1.pressure, p2.pressure, p3.pressure).clamp(0.0, 1.0),
        tilt_x: cr(p0.tilt_x, p1.tilt_x, p2.tilt_x, p3.tilt_x),
        tilt_y: cr(p0.tilt_y, p1.tilt_y, p2.tilt_y, p3.tilt_y),
        rotation: cr(p0.rotation, p1.rotation, p2.rotation, p3.rotation),
        velocity: cr(p0.velocity, p1.velocity, p2.velocity, p3.velocity).max(0.0),
        timestamp: cr(p0.timestamp, p1.timestamp, p2.timestamp, p3.timestamp),
    }
}

/// Interpolate evenly-spaced dab positions along a path.
///
/// `spacing_px` is the distance in pixels between consecutive dabs.
/// Returns dab positions with interpolated stroke point data.
pub fn spacing_interpolation(path: &[StrokePoint], spacing_px: f32) -> Vec<StrokePoint> {
    if path.is_empty() || spacing_px <= 0.0 {
        return vec![];
    }
    if path.len() == 1 {
        return vec![path[0]];
    }

    let mut result = vec![path[0]];
    let mut residual = 0.0f32; // distance remaining from last dab

    for i in 0..path.len() - 1 {
        let a = &path[i];
        let b = &path[i + 1];
        let seg_len = a.dist(b);
        if seg_len < 1e-6 {
            continue;
        }

        let mut traveled = residual;
        while traveled + spacing_px <= seg_len {
            traveled += spacing_px;
            let t = traveled / seg_len;
            result.push(StrokePoint::lerp(a, b, t));
        }
        residual = seg_len - traveled;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(x: f32, y: f32) -> StrokePoint {
        StrokePoint {
            x,
            y,
            pressure: 1.0,
            tilt_x: 0.0,
            tilt_y: 0.0,
            rotation: 0.0,
            velocity: 0.0,
            timestamp: 0.0,
        }
    }

    #[test]
    fn spacing_straight_line() {
        let path = vec![make_point(0.0, 0.0), make_point(100.0, 0.0)];
        let dabs = spacing_interpolation(&path, 10.0);
        // 0, 10, 20, ..., 100 → 11 points? Actually: 0 + (100/10) = 10 steps + initial = 11
        assert_eq!(dabs.len(), 11);
        assert!((dabs[0].x - 0.0).abs() < 1e-4);
        assert!((dabs[5].x - 50.0).abs() < 1e-4);
        assert!((dabs[10].x - 100.0).abs() < 1e-4);
    }

    #[test]
    fn spacing_non_divisible() {
        let path = vec![make_point(0.0, 0.0), make_point(25.0, 0.0)];
        let dabs = spacing_interpolation(&path, 10.0);
        // 0, 10, 20 — then 5px left (< spacing), so no dab at 25
        assert_eq!(dabs.len(), 3);
    }

    #[test]
    fn spacing_multi_segment() {
        let path = vec![
            make_point(0.0, 0.0),
            make_point(50.0, 0.0),
            make_point(100.0, 0.0),
        ];
        let dabs = spacing_interpolation(&path, 10.0);
        assert_eq!(dabs.len(), 11); // continuous spacing across segments
    }

    #[test]
    fn smooth_preserves_endpoints() {
        let path = vec![
            make_point(0.0, 0.0),
            make_point(50.0, 50.0),
            make_point(100.0, 0.0),
        ];
        let smoothed = smooth_catmull_rom(&path, 1.0, 4);
        assert!((smoothed[0].x - 0.0).abs() < 1e-4);
        let last = smoothed.last().unwrap();
        assert!((last.x - 100.0).abs() < 1e-4);
    }

    #[test]
    fn spacing_interpolates_pressure() {
        let mut a = make_point(0.0, 0.0);
        a.pressure = 0.0;
        let mut b = make_point(100.0, 0.0);
        b.pressure = 1.0;
        let dabs = spacing_interpolation(&[a, b], 50.0);
        assert_eq!(dabs.len(), 3); // 0, 50, 100
        assert!((dabs[0].pressure - 0.0).abs() < 1e-4);
        assert!((dabs[1].pressure - 0.5).abs() < 1e-4);
        assert!((dabs[2].pressure - 1.0).abs() < 1e-4);
    }
}
