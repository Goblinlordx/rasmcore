//! Filter: simplex_noise (category: generator)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a Simplex noise image (Gray8).
///
/// On WASM: f32 arithmetic (SIMD-friendly, verified u8-identical to f64).
/// On native: f64 scalar (LLVM auto-vectorizes to SSE/NEON).
#[rasmcore_macros::register_generator(
    name = "simplex_noise",
    category = "generator",
    group = "noise_gen",
    variant = "simplex",
    reference = "Perlin 2001 simplex noise"
)]
pub fn simplex_noise(width: u32, height: u32, seed: u64, scale: f64, octaves: u32) -> Vec<u8> {
    let perm = build_perm_table(seed);
    let octaves = octaves.clamp(1, 16);
    let mut pixels = vec![0u8; (width * height) as usize];

    #[cfg(target_arch = "wasm32")]
    {
        let scale = scale as f32;
        for y in 0..height {
            for x in 0..width {
                let n = fbm_f32(
                    &perm,
                    x as f32 * scale,
                    y as f32 * scale,
                    octaves,
                    simplex_2d_f32,
                );
                pixels[(y * width + x) as usize] =
                    ((n * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    for y in 0..height {
        for x in 0..width {
            let nx = x as f64 * scale;
            let ny = y as f64 * scale;
            let n = fbm(
                |fx, fy| simplex_2d(&perm, fx, fy),
                nx,
                ny,
                octaves,
                2.0,
                0.5,
            );
            pixels[(y * width + x) as usize] = ((n * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    pixels
}
