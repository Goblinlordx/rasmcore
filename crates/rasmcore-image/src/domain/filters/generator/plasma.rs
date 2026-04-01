//! Filter: plasma (category: generator)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a plasma fractal noise image via diamond-square algorithm.
///
/// Produces colorful fractal patterns. Deterministic for a given seed.
/// IM equivalent: `magick -size WxH plasma:`
#[rasmcore_macros::register_generator(
    name = "plasma",
    category = "generator",
    reference = "diamond-square fractal plasma pattern"
)]
pub fn plasma(width: u32, height: u32, seed: u64, turbulence: f32) -> Vec<u8> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let turbulence = turbulence.clamp(0.1, 10.0);

    // Use diamond-square on a power-of-2 grid, then sample
    let size = w.max(h).next_power_of_two() + 1;
    let mut grid = vec![0.0f32; size * size];

    // Simple LCG seeded random
    let mut rng_state = seed;
    let mut rng = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
    };

    // Seed corners
    grid[0] = rng();
    grid[size - 1] = rng();
    grid[(size - 1) * size] = rng();
    grid[(size - 1) * size + size - 1] = rng();

    let mut step = size - 1;
    let mut scale = turbulence;

    while step > 1 {
        let half = step / 2;

        // Diamond step
        for y in (0..size - 1).step_by(step) {
            for x in (0..size - 1).step_by(step) {
                let avg = (grid[y * size + x]
                    + grid[y * size + x + step]
                    + grid[(y + step) * size + x]
                    + grid[(y + step) * size + x + step])
                    / 4.0;
                grid[(y + half) * size + x + half] = avg + rng() * scale;
            }
        }

        // Square step
        for y in (0..size).step_by(half) {
            let x_start = if (y / half).is_multiple_of(2) {
                half
            } else {
                0
            };
            for x in (x_start..size).step_by(step) {
                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                if y >= half {
                    sum += grid[(y - half) * size + x];
                    count += 1.0;
                }
                if y + half < size {
                    sum += grid[(y + half) * size + x];
                    count += 1.0;
                }
                if x >= half {
                    sum += grid[y * size + x - half];
                    count += 1.0;
                }
                if x + half < size {
                    sum += grid[y * size + x + half];
                    count += 1.0;
                }
                grid[y * size + x] = sum / count + rng() * scale;
            }
        }

        step = half;
        scale *= 0.5;
    }

    // Normalize grid to [0, 1]
    let min_v = grid.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = grid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_v - min_v).max(1e-6);

    // Generate RGB from normalized values using a simple color mapping
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let gx = (x as f32 / w as f32 * (size - 1) as f32) as usize;
            let gy = (y as f32 / h as f32 * (size - 1) as f32) as usize;
            let t = (grid[gy.min(size - 1) * size + gx.min(size - 1)] - min_v) / range;

            // Map to a colorful gradient: blue -> cyan -> green -> yellow -> red
            let (r, g, b) = if t < 0.25 {
                let s = t / 0.25;
                (0.0, s, 1.0)
            } else if t < 0.5 {
                let s = (t - 0.25) / 0.25;
                (0.0, 1.0, 1.0 - s)
            } else if t < 0.75 {
                let s = (t - 0.5) / 0.25;
                (s, 1.0, 0.0)
            } else {
                let s = (t - 0.75) / 0.25;
                (1.0, 1.0 - s, 0.0)
            };

            let idx = (y * w + x) * 3;
            pixels[idx] = (r * 255.0 + 0.5) as u8;
            pixels[idx + 1] = (g * 255.0 + 0.5) as u8;
            pixels[idx + 2] = (b * 255.0 + 0.5) as u8;
        }
    }
    pixels
}
