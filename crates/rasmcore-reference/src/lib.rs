//! Textbook reference implementations for filter parity testing.
//!
//! This crate contains dead-simple, unoptimized reference implementations
//! of every filter operation. Pure math — no SIMD, no GPU, no optimization.
//! Each function is independently verified against a production tool
//! (see VALIDATION.md for per-filter audit trail).
//!
//! # Color Space
//!
//! All operations assume **linear f32** input, matching the rasmcore V2
//! pipeline working space. External tool validation uses linear-space
//! processing (e.g., ImageMagick `-colorspace Linear`).
//!
//! # Usage
//!
//! ```ignore
//! let input = rasmcore_reference::gradient(32, 32);
//! let expected = rasmcore_reference::point_ops::brightness(&input, 32, 32, 0.3);
//! // Compare against your optimized implementation
//! ```

pub mod color_ops;
pub mod color_ops2;
pub mod distortion_ops;
pub mod effect_ops;
pub mod enhancement_ops2;
pub mod point_ops;
pub mod spatial_ops2;

// ─── Procedural Test Input Generators ───────────────────────────────────────

/// Generate a gradient test image. Deterministic, no randomness.
///
/// Pixel at (x,y): R=x/(w-1), G=y/(h-1), B=(x+y)/(w+h-2), A=1.0
/// For w=1 or h=1, the respective channel is 0.0.
pub fn gradient(w: u32, h: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    let dw = if w > 1 { (w - 1) as f32 } else { 1.0 };
    let dh = if h > 1 { (h - 1) as f32 } else { 1.0 };
    let db = if w + h > 2 { (w + h - 2) as f32 } else { 1.0 };
    for y in 0..h {
        for x in 0..w {
            pixels.push(x as f32 / dw);
            pixels.push(y as f32 / dh);
            pixels.push((x + y) as f32 / db);
            pixels.push(1.0);
        }
    }
    pixels
}

/// Generate a solid color test image.
pub fn solid(w: u32, h: u32, color: [f32; 4]) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut pixels = Vec::with_capacity(n * 4);
    for _ in 0..n {
        pixels.extend_from_slice(&color);
    }
    pixels
}

/// Generate a deterministic pseudo-random noise image.
///
/// Uses a simple hash function — NOT cryptographic, just deterministic.
pub fn noise(w: u32, h: u32, seed: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for i in 0..(w * h) {
        let s = seed.wrapping_add(i);
        let r = hash_f32(s);
        let g = hash_f32(s.wrapping_add(1));
        let b = hash_f32(s.wrapping_add(2));
        pixels.extend_from_slice(&[r, g, b, 1.0]);
    }
    pixels
}

fn hash_f32(mut x: u32) -> f32 {
    x = x.wrapping_mul(0x9e3779b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85ebca6b);
    x ^= x >> 13;
    (x & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

/// Compare two pixel buffers and return max absolute difference.
pub fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Assert pixel buffers match within tolerance.
pub fn assert_parity(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    let diff = max_diff(actual, expected);
    assert!(
        diff <= tolerance,
        "PARITY FAIL [{name}]: max diff {diff:.10} > tolerance {tolerance:.10}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_deterministic() {
        let a = gradient(16, 16);
        let b = gradient(16, 16);
        assert_eq!(a, b);
    }

    #[test]
    fn gradient_corners() {
        let g = gradient(4, 4);
        // (0,0): R=0, G=0, B=0
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 0.0);
        assert_eq!(g[2], 0.0);
        // (3,3): R=1, G=1, B=1
        let last = (3 * 4 + 3) * 4;
        assert!((g[last] - 1.0).abs() < 1e-6);
        assert!((g[last + 1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn noise_deterministic() {
        let a = noise(8, 8, 42);
        let b = noise(8, 8, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn noise_different_seeds_differ() {
        let a = noise(8, 8, 1);
        let b = noise(8, 8, 2);
        assert_ne!(a, b);
    }
}
