//! Brush engine types — stroke points, brush params, dab instances.

/// A single point on the stroke path (matches WIT stroke-point record).
#[derive(Debug, Clone, Copy)]
pub struct StrokePoint {
    pub x: f32,
    pub y: f32,
    pub pressure: f32,
    pub tilt_x: f32,
    pub tilt_y: f32,
    pub rotation: f32,
    pub velocity: f32,
    pub timestamp: f32,
}

impl StrokePoint {
    /// Linear interpolation between two points.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let l = |a: f32, b: f32| a + t * (b - a);
        Self {
            x: l(a.x, b.x),
            y: l(a.y, b.y),
            pressure: l(a.pressure, b.pressure),
            tilt_x: l(a.tilt_x, b.tilt_x),
            tilt_y: l(a.tilt_y, b.tilt_y),
            rotation: l(a.rotation, b.rotation),
            velocity: l(a.velocity, b.velocity),
            timestamp: l(a.timestamp, b.timestamp),
        }
    }

    pub fn dist(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Brush parameters (matches WIT brush-params record).
#[derive(Debug, Clone)]
pub struct BrushParams {
    /// Base diameter in pixels.
    pub diameter: f32,
    /// Spacing as fraction of diameter (0.05 = 5% of diameter between dabs).
    pub spacing: f32,
    /// Hardness: 0.0 = soft edge, 1.0 = hard edge.
    pub hardness: f32,
    /// Flow: opacity per dab stamp (builds up with overlap).
    pub flow: f32,
    /// Stroke opacity: final composite opacity.
    pub opacity: f32,
    /// Base angle in radians.
    pub angle: f32,
    /// Roundness: 1.0 = circle, 0.0 = flat line.
    pub roundness: f32,
    /// Scatter: random perpendicular displacement (fraction of diameter).
    pub scatter: f32,
    /// Smoothing: Catmull-Rom tension (0 = no smoothing, 1 = max).
    pub smoothing: f32,
    /// Dynamics curves: maps input (pressure, velocity, tilt) to size/opacity/angle.
    pub dynamics: DynamicsCurves,
}

impl Default for BrushParams {
    fn default() -> Self {
        Self {
            diameter: 20.0,
            spacing: 0.15,
            hardness: 0.8,
            flow: 1.0,
            opacity: 1.0,
            angle: 0.0,
            roundness: 1.0,
            scatter: 0.0,
            smoothing: 0.5,
            dynamics: DynamicsCurves::default(),
        }
    }
}

/// Dynamics curves — each maps an input property to a multiplier [0, 1].
#[derive(Debug, Clone, Default)]
pub struct DynamicsCurves {
    /// Pressure → size multiplier.
    pub pressure_size: DynamicsCurve,
    /// Pressure → opacity multiplier.
    pub pressure_opacity: DynamicsCurve,
    /// Velocity → size multiplier.
    pub velocity_size: DynamicsCurve,
    /// Tilt → angle offset.
    pub tilt_angle: DynamicsCurve,
}

/// A piecewise-linear curve mapping input [0,1] → output [0,1].
/// Empty = identity (output = 1.0 always).
#[derive(Debug, Clone)]
pub struct DynamicsCurve {
    /// Sorted control points (input, output). Input in [0, 1].
    pub points: Vec<(f32, f32)>,
}

impl Default for DynamicsCurve {
    fn default() -> Self {
        // Default: identity (full effect at any input)
        Self { points: vec![] }
    }
}

/// A single dab instance — the result of evaluating dynamics at a path position.
#[derive(Debug, Clone, Copy)]
pub struct DabInstance {
    /// Center position in pixels.
    pub x: f32,
    pub y: f32,
    /// Diameter in pixels (after dynamics).
    pub size: f32,
    /// Opacity (flow * dynamics modifier).
    pub opacity: f32,
    /// Angle in radians (after dynamics).
    pub angle: f32,
    /// Roundness (from brush params).
    pub roundness: f32,
    /// Hardness (from brush params).
    pub hardness: f32,
}

/// f32 RGBA accumulation buffer for stroke rendering.
pub struct AccumulationBuffer {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl AccumulationBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0.0; (width * height * 4) as usize],
            width,
            height,
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }
}
