//! Dynamics curve evaluation — piecewise-linear lookup.

use super::types::DynamicsCurve;

impl DynamicsCurve {
    /// Evaluate the curve at input value t ∈ [0, 1].
    /// Returns output value ∈ [0, 1].
    /// Empty curve = always returns 1.0 (no modulation).
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        if self.points.is_empty() {
            return 1.0;
        }
        if self.points.len() == 1 {
            return self.points[0].1;
        }

        // Before first point
        if t <= self.points[0].0 {
            return self.points[0].1;
        }
        // After last point
        if t >= self.points[self.points.len() - 1].0 {
            return self.points[self.points.len() - 1].1;
        }

        // Find segment and interpolate
        for i in 0..self.points.len() - 1 {
            let (x0, y0) = self.points[i];
            let (x1, y1) = self.points[i + 1];
            if t >= x0 && t <= x1 {
                let seg_t = if (x1 - x0).abs() < 1e-10 {
                    0.0
                } else {
                    (t - x0) / (x1 - x0)
                };
                return y0 + seg_t * (y1 - y0);
            }
        }

        // Fallback
        self.points.last().unwrap().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_curve_returns_one() {
        let c = DynamicsCurve::default();
        assert_eq!(c.evaluate(0.0), 1.0);
        assert_eq!(c.evaluate(0.5), 1.0);
        assert_eq!(c.evaluate(1.0), 1.0);
    }

    #[test]
    fn identity_curve() {
        let c = DynamicsCurve {
            points: vec![(0.0, 0.0), (1.0, 1.0)],
        };
        assert!((c.evaluate(0.0) - 0.0).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inverted_curve() {
        let c = DynamicsCurve {
            points: vec![(0.0, 1.0), (1.0, 0.0)],
        };
        assert!((c.evaluate(0.0) - 1.0).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn multi_segment_curve() {
        let c = DynamicsCurve {
            points: vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.5)],
        };
        assert!((c.evaluate(0.25) - 0.5).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 1.0).abs() < 1e-6);
        assert!((c.evaluate(0.75) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn clamps_out_of_range() {
        let c = DynamicsCurve {
            points: vec![(0.2, 0.3), (0.8, 0.9)],
        };
        assert_eq!(c.evaluate(0.0), 0.3);
        assert_eq!(c.evaluate(1.0), 0.9);
    }
}
