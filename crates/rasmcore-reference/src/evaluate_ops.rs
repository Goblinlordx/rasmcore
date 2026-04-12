//! Per-pixel evaluate operations — trivial math on each RGB channel.
//!
//! All operations work in **linear f32** space (RGBA, 4 channels interleaved).
//! Alpha is preserved unchanged. Pure math — no SIMD, no GPU, no external crates.
//!
//! Each function takes `&[f32]` RGBA input and returns a new `Vec<f32>`.

// ─── Evaluate Operations ──────────────────────────────────────────────────────

/// Absolute value of each RGB channel.
///
/// Formula: `out = |channel|`
///
/// Validated against: ImageMagick `-evaluate Abs 0`
pub fn evaluate_abs(input: &[f32], _w: u32, _h: u32, _value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[0] = px[0].abs();
        px[1] = px[1].abs();
        px[2] = px[2].abs();
        // alpha unchanged
    }
    out
}

/// Add a constant to each RGB channel.
///
/// Formula: `out = channel + value`
///
/// Validated against: ImageMagick `-evaluate Add {value}`
pub fn evaluate_add(input: &[f32], _w: u32, _h: u32, value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[0] += value;
        px[1] += value;
        px[2] += value;
    }
    out
}

/// Subtract a constant from each RGB channel.
///
/// Formula: `out = channel - value`
///
/// Validated against: ImageMagick `-evaluate Subtract {value}`
pub fn evaluate_subtract(input: &[f32], _w: u32, _h: u32, value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[0] -= value;
        px[1] -= value;
        px[2] -= value;
    }
    out
}

/// Multiply each RGB channel by a constant.
///
/// Formula: `out = channel * value`
///
/// Validated against: ImageMagick `-evaluate Multiply {value}`
pub fn evaluate_multiply(input: &[f32], _w: u32, _h: u32, value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[0] *= value;
        px[1] *= value;
        px[2] *= value;
    }
    out
}

/// Divide each RGB channel by a constant.
///
/// Formula: `out = channel / value` (if value ≈ 0, return 0)
///
/// Validated against: ImageMagick `-evaluate Divide {value}`
pub fn evaluate_divide(input: &[f32], _w: u32, _h: u32, value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    if value.abs() < 1e-10 {
        // Safe: division by zero → all RGB channels become 0
        for px in out.chunks_exact_mut(4) {
            px[0] = 0.0;
            px[1] = 0.0;
            px[2] = 0.0;
        }
    } else {
        for px in out.chunks_exact_mut(4) {
            px[0] /= value;
            px[1] /= value;
            px[2] /= value;
        }
    }
    out
}

/// Raise each RGB channel to a power.
///
/// Formula: `out = channel.powf(value)` (for channel > 0, else 0)
///
/// Validated against: ImageMagick `-evaluate Pow {value}`
pub fn evaluate_pow(input: &[f32], _w: u32, _h: u32, value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            px[c] = if px[c] > 0.0 { px[c].powf(value) } else { 0.0 };
        }
    }
    out
}

/// Natural logarithm with offset to avoid log(0).
///
/// Formula: `out = ln(channel * value + 1)`
///
/// The +1 offset ensures the argument is always >= 1 when channel >= 0
/// and value >= 0, avoiding log of zero or negative numbers.
///
/// Validated against: numpy ln(1 + max(pixel, 0)) * scale
///
/// Formula: `out = ln(1 + max(channel, 0)) * scale`
pub fn evaluate_log(input: &[f32], _w: u32, _h: u32, scale: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            px[c] = (1.0 + px[c].max(0.0)).ln() * scale;
        }
    }
    out
}

/// Per-channel maximum of the channel value and a constant.
///
/// Formula: `out = max(channel, value)`
///
/// Validated against: ImageMagick `-evaluate Max {value}`
pub fn evaluate_max(input: &[f32], _w: u32, _h: u32, value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[0] = px[0].max(value);
        px[1] = px[1].max(value);
        px[2] = px[2].max(value);
    }
    out
}

/// Per-channel minimum of the channel value and a constant.
///
/// Formula: `out = min(channel, value)`
///
/// Validated against: ImageMagick `-evaluate Min {value}`
pub fn evaluate_min(input: &[f32], _w: u32, _h: u32, value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[0] = px[0].min(value);
        px[1] = px[1].min(value);
        px[2] = px[2].min(value);
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn add_zero_is_identity() {
        let input = crate::gradient(16, 16);
        let result = evaluate_add(&input, 16, 16, 0.0);
        assert!(max_diff(&input, &result) < TOL, "add(0) must be identity");
    }

    #[test]
    fn multiply_one_is_identity() {
        let input = crate::gradient(16, 16);
        let result = evaluate_multiply(&input, 16, 16, 1.0);
        assert!(max_diff(&input, &result) < TOL, "multiply(1) must be identity");
    }

    #[test]
    fn abs_on_positive_is_identity() {
        let input = crate::gradient(16, 16);
        let result = evaluate_abs(&input, 16, 16, 0.0);
        assert!(
            max_diff(&input, &result) < TOL,
            "abs on non-negative input must be identity"
        );
    }

    #[test]
    fn subtract_then_add_roundtrip() {
        let input = crate::gradient(8, 8);
        let sub = evaluate_subtract(&input, 8, 8, 0.3);
        let roundtrip = evaluate_add(&sub, 8, 8, 0.3);
        assert!(
            max_diff(&input, &roundtrip) < TOL,
            "subtract then add same value must roundtrip"
        );
    }

    #[test]
    fn multiply_then_divide_roundtrip() {
        let input = crate::solid(4, 4, [0.5, 0.3, 0.7, 1.0]);
        let mul = evaluate_multiply(&input, 4, 4, 2.0);
        let roundtrip = evaluate_divide(&mul, 4, 4, 2.0);
        assert!(
            max_diff(&input, &roundtrip) < TOL,
            "multiply then divide by same value must roundtrip"
        );
    }

    #[test]
    fn divide_by_zero_returns_zero() {
        let input = crate::gradient(4, 4);
        let result = evaluate_divide(&input, 4, 4, 0.0);
        for px in result.chunks_exact(4) {
            assert_eq!(px[0], 0.0);
            assert_eq!(px[1], 0.0);
            assert_eq!(px[2], 0.0);
        }
    }

    #[test]
    fn pow_preserves_zero() {
        let input = crate::solid(4, 4, [0.0, 0.0, 0.0, 1.0]);
        let result = evaluate_pow(&input, 4, 4, 2.0);
        for px in result.chunks_exact(4) {
            assert_eq!(px[0], 0.0);
            assert_eq!(px[1], 0.0);
            assert_eq!(px[2], 0.0);
        }
    }

    #[test]
    fn log_of_zero_value_is_zero() {
        // ln(channel * 0 + 1) = ln(1) = 0
        let input = crate::gradient(4, 4);
        let result = evaluate_log(&input, 4, 4, 0.0);
        for px in result.chunks_exact(4) {
            assert!(px[0].abs() < TOL);
            assert!(px[1].abs() < TOL);
            assert!(px[2].abs() < TOL);
        }
    }

    #[test]
    fn max_clamps_up() {
        let input = crate::solid(4, 4, [0.2, 0.3, 0.4, 1.0]);
        let result = evaluate_max(&input, 4, 4, 0.5);
        for px in result.chunks_exact(4) {
            assert!((px[0] - 0.5).abs() < TOL);
            assert!((px[1] - 0.5).abs() < TOL);
            assert!((px[2] - 0.5).abs() < TOL);
        }
    }

    #[test]
    fn min_clamps_down() {
        let input = crate::solid(4, 4, [0.8, 0.9, 0.7, 1.0]);
        let result = evaluate_min(&input, 4, 4, 0.5);
        for px in result.chunks_exact(4) {
            assert!((px[0] - 0.5).abs() < TOL);
            assert!((px[1] - 0.5).abs() < TOL);
            assert!((px[2] - 0.5).abs() < TOL);
        }
    }

    #[test]
    fn all_ops_preserve_alpha() {
        let input = crate::solid(2, 2, [0.5, 0.5, 0.5, 0.75]);
        let ops: Vec<(&str, Vec<f32>)> = vec![
            ("abs", evaluate_abs(&input, 2, 2, 0.0)),
            ("add", evaluate_add(&input, 2, 2, 0.1)),
            ("sub", evaluate_subtract(&input, 2, 2, 0.1)),
            ("mul", evaluate_multiply(&input, 2, 2, 2.0)),
            ("div", evaluate_divide(&input, 2, 2, 2.0)),
            ("pow", evaluate_pow(&input, 2, 2, 2.0)),
            ("log", evaluate_log(&input, 2, 2, 1.0)),
            ("max", evaluate_max(&input, 2, 2, 0.3)),
            ("min", evaluate_min(&input, 2, 2, 0.8)),
        ];
        for (name, result) in &ops {
            for px in result.chunks_exact(4) {
                assert!(
                    (px[3] - 0.75).abs() < TOL,
                    "{name} must preserve alpha, got {}",
                    px[3]
                );
            }
        }
    }
}
