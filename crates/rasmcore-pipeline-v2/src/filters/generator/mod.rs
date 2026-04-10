//! Generator filters — procedural pattern generation.
//!
//! These replace the input image with a generated pattern. All are pure
//! per-pixel math — ideal for GPU compute shaders.

mod checkerboard;
mod cloud_noise;
mod fractal_noise;
mod gradient_linear;
mod gradient_radial;
mod pattern_fill;
mod perlin_noise;
mod plasma;
mod simplex_noise;
mod solid_color;

pub use checkerboard::Checkerboard;
pub use cloud_noise::CloudNoise;
pub use fractal_noise::FractalNoise;
pub use gradient_linear::GradientLinear;
pub use gradient_radial::GradientRadial;
pub use pattern_fill::PatternFill;
pub use perlin_noise::PerlinNoise;
pub use plasma::Plasma;
pub use simplex_noise::SimplexNoise;
pub use solid_color::SolidColor;

// ─── CPU noise helpers ─────────────────────────────────────────────────────

pub(super) fn hash2_cpu(x: f32, y: f32) -> f32 {
    let kx = 0.3183099f32;
    let ky = 0.3678794f32;
    let px = x * kx + ky;
    let py = y * ky + kx;
    (16.0 * kx * (px * py * (px + py)).fract()).fract()
}

pub(super) fn noise2_cpu(px: f32, py: f32) -> f32 {
    let ix = px.floor();
    let iy = py.floor();
    let fx = px - ix;
    let fy = py - iy;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash2_cpu(ix, iy);
    let b = hash2_cpu(ix + 1.0, iy);
    let c = hash2_cpu(ix, iy + 1.0);
    let d = hash2_cpu(ix + 1.0, iy + 1.0);
    let ab = a + ux * (b - a);
    let cd = c + ux * (d - c);
    ab + uy * (cd - ab)
}

pub(super) fn fbm_cpu(x: f32, y: f32, octaves: u32, persistence: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut total = 0.0f32;
    for _ in 0..octaves {
        value += noise2_cpu(x * frequency, y * frequency) * amplitude;
        total += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    value / total
}

// ─── CPU noise helpers (Worley + lacunarity) ──────────────────────────────

pub(super) fn hash2v_cpu(x: f32, y: f32) -> (f32, f32) {
    (hash2_cpu(x, y), hash2_cpu(x + 127.1, y + 311.7))
}

pub(super) fn worley_cpu(px: f32, py: f32) -> f32 {
    let ix = px.floor();
    let iy = py.floor();
    let fx = px - ix;
    let fy = py - iy;
    let mut min_dist = 1.0f32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (hx, hy) = hash2v_cpu(ix + dx as f32, iy + dy as f32);
            let diff_x = dx as f32 + hx - fx;
            let diff_y = dy as f32 + hy - fy;
            let d = (diff_x * diff_x + diff_y * diff_y).sqrt();
            min_dist = min_dist.min(d);
        }
    }
    min_dist
}

pub(super) fn worley_fbm_cpu(x: f32, y: f32, octaves: u32, persistence: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut total = 0.0f32;
    for _ in 0..octaves {
        value += (1.0 - worley_cpu(x * frequency, y * frequency)) * amplitude;
        total += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    value / total
}

pub(super) fn fbm_lacunarity_cpu(
    x: f32,
    y: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut total = 0.0f32;
    for _ in 0..octaves {
        value += noise2_cpu(x * frequency, y * frequency) * amplitude;
        total += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    value / total
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

    #[test]
    fn all_generator_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &[
            "checkerboard",
            "gradient_linear",
            "gradient_radial",
            "perlin_noise",
            "simplex_noise",
            "plasma",
            "solid_color",
            "fractal_noise",
            "cloud_noise",
            "pattern_fill",
        ] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn checkerboard_produces_two_colors() {
        let input = vec![0.0f32; 64 * 64 * 4];
        let f = Checkerboard {
            size: 32.0,
            color1_r: 1.0,
            color1_g: 1.0,
            color1_b: 1.0,
            color2_r: 0.0,
            color2_g: 0.0,
            color2_b: 0.0,
        };
        let out = f.compute(&input, 64, 64).unwrap();
        // (0,0) should be color1 (white)
        assert!(out[0] > 0.9);
        // (32,0) should be color2 (black)
        let i = (0 * 64 + 32) * 4;
        assert!(out[i] < 0.1);
    }

    #[test]
    fn plasma_produces_color() {
        let input = vec![0.0f32; 32 * 32 * 4];
        let f = Plasma {
            scale: 10.0,
            time: 0.0,
        };
        let out = f.compute(&input, 32, 32).unwrap();
        let has_color = out
            .chunks(4)
            .any(|px| px[0] > 0.01 || px[1] > 0.01 || px[2] > 0.01);
        assert!(has_color, "plasma should produce visible color");
    }

    #[test]
    fn solid_color_fills_constant() {
        let input = vec![0.0f32; 8 * 8 * 4];
        let f = SolidColor {
            r: 0.3,
            g: 0.6,
            b: 0.9,
            a: 1.0,
        };
        let out = f.compute(&input, 8, 8).unwrap();
        // Every pixel should be the same
        for px in out.chunks(4) {
            assert!((px[0] - 0.3).abs() < 1e-6);
            assert!((px[1] - 0.6).abs() < 1e-6);
            assert!((px[2] - 0.9).abs() < 1e-6);
            assert!((px[3] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn fractal_noise_deterministic() {
        let input = vec![0.0f32; 32 * 32 * 4];
        let f = FractalNoise {
            scale: 20.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
            seed: 42,
        };
        let out1 = f.compute(&input, 32, 32).unwrap();
        let out2 = f.compute(&input, 32, 32).unwrap();
        assert_eq!(out1, out2, "same params should produce identical output");
    }

    #[test]
    fn fractal_noise_lacunarity_differs_from_perlin() {
        let input = vec![0.0f32; 16 * 16 * 4];
        let frac = FractalNoise {
            scale: 20.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 3.0,
            seed: 42,
        };
        let perl = PerlinNoise {
            scale: 20.0,
            octaves: 4,
            persistence: 0.5,
            seed: 42,
        };
        let out_frac = frac.compute(&input, 16, 16).unwrap();
        let out_perl = perl.compute(&input, 16, 16).unwrap();
        // With lacunarity=3 vs implicit 2, results should differ
        assert_ne!(out_frac, out_perl);
    }

    #[test]
    fn cloud_noise_produces_visible_output() {
        let input = vec![0.0f32; 32 * 32 * 4];
        let f = CloudNoise {
            scale: 20.0,
            octaves: 3,
            persistence: 0.5,
            worley_blend: 0.4,
            seed: 7,
        };
        let out = f.compute(&input, 32, 32).unwrap();
        let has_visible = out.chunks(4).any(|px| px[0] > 0.01);
        assert!(has_visible, "cloud noise should produce visible output");
    }

    #[test]
    fn cloud_noise_deterministic() {
        let input = vec![0.0f32; 16 * 16 * 4];
        let f = CloudNoise {
            scale: 20.0,
            octaves: 3,
            persistence: 0.5,
            worley_blend: 0.4,
            seed: 7,
        };
        let out1 = f.compute(&input, 16, 16).unwrap();
        let out2 = f.compute(&input, 16, 16).unwrap();
        assert_eq!(out1, out2);
    }

    #[test]
    fn pattern_fill_tiles_input() {
        // Create a 4x4 image with known values, tile at 2x2
        let mut input = vec![0.0f32; 4 * 4 * 4];
        // Set pixel (0,0) to red
        input[0] = 1.0;
        input[3] = 1.0;
        // Set pixel (1,0) to green
        input[4 + 1] = 1.0;
        input[4 + 3] = 1.0;
        let f = PatternFill {
            tile_w: 2.0,
            tile_h: 2.0,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let out = f.compute(&input, 4, 4).unwrap();
        // (2,0) should match (0,0) = red
        let i = (0 * 4 + 2) * 4;
        assert!((out[i] - 1.0).abs() < 1e-6, "tiled pixel should be red");
        // (3,0) should match (1,0) = green
        let j = (0 * 4 + 3) * 4;
        assert!(
            (out[j + 1] - 1.0).abs() < 1e-6,
            "tiled pixel should be green"
        );
    }
}
