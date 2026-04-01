//! Noise helpers for filters.

#[allow(unused_imports)]
use super::*;


pub const SIMPLEX_STRETCH: f64 = -0.211324865405187; // (1/sqrt(3) - 1) / 2
pub const SIMPLEX_SQUISH: f64 = 0.366025403784439; // (sqrt(3) - 1) / 2

/// Gradient table for OpenSimplex 2D (8 directions).
pub const SIMPLEX_GRADS: [(f64, f64); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        -std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        -std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
];

/// Box-Muller transform: two uniform randoms → two standard-normal values.
/// Returns one value per call (discards the second for simplicity).
#[inline]
pub fn box_muller(state: &mut u64) -> f64 {
    let u1 = xorshift64_f64(state).max(1e-300); // avoid log(0)
    let u2 = xorshift64_f64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Build a seeded permutation table (256 entries, doubled for wrapping).
pub fn build_perm_table(seed: u64) -> [u8; 512] {
    let mut perm = [0u8; 256];
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i as u8;
    }
    // Fisher-Yates shuffle with a simple LCG seeded from the user seed
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..256).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }
    let mut table = [0u8; 512];
    for i in 0..512 {
        table[i] = perm[i & 255];
    }
    table
}

/// Fade curve: 6t^5 - 15t^4 + 10t^3 (Perlin improved noise)
#[inline]
pub fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn fade_f32(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Layer multiple octaves of noise for natural-looking results.
pub fn fbm<F>(noise_fn: F, x: f64, y: f64, octaves: u32, lacunarity: f64, persistence: f64) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let mut value = 0.0f64;
    let mut amplitude = 1.0f64;
    let mut frequency = 1.0f64;
    let mut max_amp = 0.0f64;

    for _ in 0..octaves {
        value += noise_fn(x * frequency, y * frequency) * amplitude;
        max_amp += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_amp // normalize to [-1, 1]
}

#[cfg(target_arch = "wasm32")]
pub fn fbm_f32(
    perm: &[u8; 512],
    x: f32,
    y: f32,
    octaves: u32,
    noise_fn: fn(&[u8; 512], f32, f32) -> f32,
) -> f32 {
    let (mut v, mut a, mut fr, mut ma) = (0.0f32, 1.0f32, 1.0f32, 0.0f32);
    for _ in 0..octaves {
        v += noise_fn(perm, x * fr, y * fr) * a;
        ma += a;
        a *= 0.5;
        fr *= 2.0;
    }
    v / ma
}

/// Gradient function for improved Perlin noise.
/// Uses hash to select from 12 gradient directions (Perlin 2002).
#[inline]
pub fn grad_perlin(hash: u8, x: f64, y: f64) -> f64 {
    match hash & 0x3 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        _ => -x - y,
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn grad_perlin_f32(hash: u8, x: f32, y: f32) -> f32 {
    match hash & 0x3 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        _ => -x - y,
    }
}

/// Single-octave improved Perlin noise at (x, y). Returns [-1, 1].
pub fn perlin_2d(perm: &[u8; 512], x: f64, y: f64) -> f64 {
    let xi = x.floor() as i32 & 255;
    let yi = y.floor() as i32 & 255;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let u = fade(xf);
    let v = fade(yf);

    let aa = perm[(perm[xi as usize] as i32 + yi) as usize & 511];
    let ab = perm[(perm[xi as usize] as i32 + yi + 1) as usize & 511];
    let ba = perm[(perm[(xi + 1) as usize & 255] as i32 + yi) as usize & 511];
    let bb = perm[(perm[(xi + 1) as usize & 255] as i32 + yi + 1) as usize & 511];

    lerp(
        v,
        lerp(u, grad_perlin(aa, xf, yf), grad_perlin(ba, xf - 1.0, yf)),
        lerp(
            u,
            grad_perlin(ab, xf, yf - 1.0),
            grad_perlin(bb, xf - 1.0, yf - 1.0),
        ),
    )
}

#[cfg(target_arch = "wasm32")]
pub fn perlin_2d_f32(perm: &[u8; 512], x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32 & 255;
    let yi = y.floor() as i32 & 255;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let u = fade_f32(xf);
    let v = fade_f32(yf);
    let aa = perm[(perm[xi as usize] as i32 + yi) as usize & 511];
    let ab = perm[(perm[xi as usize] as i32 + yi + 1) as usize & 511];
    let ba = perm[(perm[(xi + 1) as usize & 255] as i32 + yi) as usize & 511];
    let bb = perm[(perm[(xi + 1) as usize & 255] as i32 + yi + 1) as usize & 511];
    lerp_f32(
        v,
        lerp_f32(
            u,
            grad_perlin_f32(aa, xf, yf),
            grad_perlin_f32(ba, xf - 1.0, yf),
        ),
        lerp_f32(
            u,
            grad_perlin_f32(ab, xf, yf - 1.0),
            grad_perlin_f32(bb, xf - 1.0, yf - 1.0),
        ),
    )
}

/// Knuth's algorithm for Poisson random variates (small lambda ≤ 30).
/// For larger lambda, use normal approximation.
#[inline]
pub fn poisson_random(lambda: f64, rng: &mut u64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda > 30.0 {
        // Normal approximation: Poisson(λ) ≈ N(λ, λ)
        return (lambda + box_muller(rng) * lambda.sqrt()).max(0.0);
    }
    // Knuth's algorithm
    let l = (-lambda).exp();
    let mut k = 0.0f64;
    let mut p = 1.0f64;
    loop {
        k += 1.0;
        p *= xorshift64_f64(rng);
        if p <= l {
            return k - 1.0;
        }
    }
}

/// Single-octave OpenSimplex noise at (x, y). Returns approximately [-1, 1].
pub fn simplex_2d(perm: &[u8; 512], x: f64, y: f64) -> f64 {
    let stretch = (x + y) * SIMPLEX_STRETCH;
    let xs = x + stretch;
    let ys = y + stretch;

    let xsb = xs.floor() as i32;
    let ysb = ys.floor() as i32;

    let squish = (xsb + ysb) as f64 * SIMPLEX_SQUISH;
    let xb = xsb as f64 + squish;
    let yb = ysb as f64 + squish;

    let dx0 = x - xb;
    let dy0 = y - yb;

    let xins = xs - xsb as f64;
    let yins = ys - ysb as f64;

    let mut value = 0.0f64;

    // Contribution from (0, 0)
    let at0 = 2.0 - dx0 * dx0 - dy0 * dy0;
    if at0 > 0.0 {
        let at0 = at0 * at0;
        let gi = perm[(perm[xsb as usize & 255] as i32 + ysb) as usize & 511] as usize & 7;
        value += at0 * at0 * (SIMPLEX_GRADS[gi].0 * dx0 + SIMPLEX_GRADS[gi].1 * dy0);
    }

    // Contribution from (1, 0)
    let dx1 = dx0 - 1.0 - SIMPLEX_SQUISH;
    let dy1 = dy0 - SIMPLEX_SQUISH;
    let at1 = 2.0 - dx1 * dx1 - dy1 * dy1;
    if at1 > 0.0 {
        let at1 = at1 * at1;
        let gi = perm[(perm[(xsb + 1) as usize & 255] as i32 + ysb) as usize & 511] as usize & 7;
        value += at1 * at1 * (SIMPLEX_GRADS[gi].0 * dx1 + SIMPLEX_GRADS[gi].1 * dy1);
    }

    // Contribution from (0, 1)
    let dx2 = dx0 - SIMPLEX_SQUISH;
    let dy2 = dy0 - 1.0 - SIMPLEX_SQUISH;
    let at2 = 2.0 - dx2 * dx2 - dy2 * dy2;
    if at2 > 0.0 {
        let at2 = at2 * at2;
        let gi = perm[(perm[xsb as usize & 255] as i32 + ysb + 1) as usize & 511] as usize & 7;
        value += at2 * at2 * (SIMPLEX_GRADS[gi].0 * dx2 + SIMPLEX_GRADS[gi].1 * dy2);
    }

    // Contribution from (1, 1) — only if in the upper triangle
    if xins + yins > 1.0 {
        let dx3 = dx0 - 1.0 - 2.0 * SIMPLEX_SQUISH;
        let dy3 = dy0 - 1.0 - 2.0 * SIMPLEX_SQUISH;
        let at3 = 2.0 - dx3 * dx3 - dy3 * dy3;
        if at3 > 0.0 {
            let at3 = at3 * at3;
            let gi =
                perm[(perm[(xsb + 1) as usize & 255] as i32 + ysb + 1) as usize & 511] as usize & 7;
            value += at3 * at3 * (SIMPLEX_GRADS[gi].0 * dx3 + SIMPLEX_GRADS[gi].1 * dy3);
        }
    }

    // Scale to approximately [-1, 1]
    value * 47.0
}

#[cfg(target_arch = "wasm32")]
pub fn simplex_2d_f32(perm: &[u8; 512], x: f32, y: f32) -> f32 {
    const SS: f32 = -0.211324865;
    const SQ: f32 = 0.366025404;
    const SR2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    const GR: [(f32, f32); 8] = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (SR2, SR2),
        (-SR2, SR2),
        (SR2, -SR2),
        (-SR2, -SR2),
    ];
    let stretch = (x + y) * SS;
    let xs = x + stretch;
    let ys = y + stretch;
    let xsb = xs.floor() as i32;
    let ysb = ys.floor() as i32;
    let squish = (xsb + ysb) as f32 * SQ;
    let dx0 = x - (xsb as f32 + squish);
    let dy0 = y - (ysb as f32 + squish);
    let xins = xs - xsb as f32;
    let yins = ys - ysb as f32;
    let mut value = 0.0f32;
    let a0 = 2.0 - dx0 * dx0 - dy0 * dy0;
    if a0 > 0.0 {
        let a = a0 * a0;
        let gi = perm[(perm[xsb as usize & 255] as i32 + ysb) as usize & 511] as usize & 7;
        value += a * a * (GR[gi].0 * dx0 + GR[gi].1 * dy0);
    }
    let d1x = dx0 - 1.0 - SQ;
    let d1y = dy0 - SQ;
    let a1 = 2.0 - d1x * d1x - d1y * d1y;
    if a1 > 0.0 {
        let a = a1 * a1;
        let gi = perm[(perm[(xsb + 1) as usize & 255] as i32 + ysb) as usize & 511] as usize & 7;
        value += a * a * (GR[gi].0 * d1x + GR[gi].1 * d1y);
    }
    let d2x = dx0 - SQ;
    let d2y = dy0 - 1.0 - SQ;
    let a2 = 2.0 - d2x * d2x - d2y * d2y;
    if a2 > 0.0 {
        let a = a2 * a2;
        let gi = perm[(perm[xsb as usize & 255] as i32 + ysb + 1) as usize & 511] as usize & 7;
        value += a * a * (GR[gi].0 * d2x + GR[gi].1 * d2y);
    }
    if xins + yins > 1.0 {
        let d3x = dx0 - 1.0 - 2.0 * SQ;
        let d3y = dy0 - 1.0 - 2.0 * SQ;
        let a3 = 2.0 - d3x * d3x - d3y * d3y;
        if a3 > 0.0 {
            let a = a3 * a3;
            let gi =
                perm[(perm[(xsb + 1) as usize & 255] as i32 + ysb + 1) as usize & 511] as usize & 7;
            value += a * a * (GR[gi].0 * d3x + GR[gi].1 * d3y);
        }
    }
    value * 47.0
}

/// xorshift64 PRNG — fast, deterministic, good enough for noise generation.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Return a uniform f64 in [0, 1) from the PRNG.
#[inline]
pub fn xorshift64_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

