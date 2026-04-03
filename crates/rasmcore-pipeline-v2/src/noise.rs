//! Deterministic noise — SplitMix64-based PRNG and spatial hash.
//!
//! All filters needing deterministic noise should use this module.
//! Provides both CPU functions and a WGSL snippet for GPU shaders.
//!
//! # Algorithm
//!
//! SplitMix64 (Steele, Lea, Flood 2014) — excellent avalanche properties,
//! passes BigCrush. Used in two modes:
//! - **Sequential** (`Rng`): state-advancing PRNG for per-pixel noise
//! - **Spatial** (`noise_2d`): hash from (x, y, seed) for position-dependent noise
//!
//! # Seed offsets
//!
//! Each algorithm type has a unique offset constant (`SEED_*`) mixed into the
//! user-provided seed. This ensures two filters with the same user seed produce
//! uncorrelated noise streams.

// ─── Seed Offsets ──────────────────────────────────────────────────────────

/// Seed offset for film grain.
pub const SEED_FILM_GRAIN: u64 = 0xA5A5_A5A5_0001_0001;
/// Seed offset for Gaussian noise.
pub const SEED_GAUSSIAN_NOISE: u64 = 0xA5A5_A5A5_0002_0002;
/// Seed offset for uniform noise.
pub const SEED_UNIFORM_NOISE: u64 = 0xA5A5_A5A5_0003_0003;
/// Seed offset for salt & pepper noise.
pub const SEED_SALT_PEPPER: u64 = 0xA5A5_A5A5_0004_0004;
/// Seed offset for Poisson noise.
pub const SEED_POISSON_NOISE: u64 = 0xA5A5_A5A5_0005_0005;
/// Seed offset for glitch effect.
pub const SEED_GLITCH: u64 = 0xA5A5_A5A5_0006_0006;
/// Seed offset for dithering.
pub const SEED_DITHER: u64 = 0xA5A5_A5A5_0007_0007;
/// Seed offset for k-means initialization.
pub const SEED_KMEANS: u64 = 0xA5A5_A5A5_0008_0008;

// ─── SplitMix64 Sequential PRNG ───────────────────────────────────────────

/// SplitMix64 PRNG — deterministic, fast, high-quality.
///
/// Use for sequential noise generation (per-pixel iteration).
/// For position-dependent noise, use `noise_2d()` instead.
pub struct Rng(u64);

impl Rng {
    /// Create a new PRNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Create a PRNG with user seed + algorithm-specific offset.
    pub fn with_offset(user_seed: u64, algo_offset: u64) -> Self {
        Self(user_seed ^ algo_offset)
    }

    /// Advance state and return next u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform f32 in [-1, 1].
    #[inline]
    pub fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Gaussian via Box-Muller transform.
    #[inline]
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

// ─── SplitMix64 Spatial Hash ───────────────────────────────────────────────

/// One-shot SplitMix64: hash a u64 state to a u64 result.
#[inline]
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 2D spatial noise in [-1, 1] using SplitMix64.
///
/// Deterministic: same (x, y, seed) always produces the same value.
/// The seed should include an algorithm offset (e.g., `user_seed ^ SEED_FILM_GRAIN`).
#[inline]
pub fn noise_2d(x: u32, y: u32, seed: u64) -> f32 {
    let state = (x as u64) | ((y as u64) << 32);
    let h = splitmix64(state ^ seed);
    (h >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

/// 1D noise in [-1, 1] using SplitMix64.
#[inline]
pub fn noise_1d(x: u32, seed: u64) -> f32 {
    let h = splitmix64(x as u64 ^ seed);
    (h >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

/// Simple seeded u32 selection (for k-means init, shuffle, etc.).
/// Returns a value in [0, max).
#[inline]
pub fn seeded_index(index: u32, seed: u64, max: u32) -> u32 {
    let h = splitmix64(index as u64 ^ seed);
    (h % max as u64) as u32
}

// ─── WGSL Snippet ──────────────────────────────────────────────────────────

/// WGSL fragment implementing SplitMix64 via emulated u64 (vec2<u32>).
///
/// Provides:
/// - `splitmix64(state: vec2<u32>) -> vec2<u32>` — one-shot hash
/// - `noise_2d(x: u32, y: u32, seed_lo: u32, seed_hi: u32) -> f32` — spatial noise in [-1, 1]
///
/// Include this in your shader body alongside io_f32.
pub const NOISE_WGSL: &str = r#"
// ─── u64 emulation via vec2<u32> (lo, hi) ──────────────────────────────────

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  let carry = select(0u, 1u, lo < a.x);
  return vec2<u32>(lo, a.y + b.y + carry);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn u64_shr(v: vec2<u32>, n: u32) -> vec2<u32> {
  if (n == 0u) { return v; }
  if (n >= 32u) { return vec2<u32>(v.y >> (n - 32u), 0u); }
  return vec2<u32>((v.x >> n) | (v.y << (32u - n)), v.y >> n);
}

fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  // 64-bit multiply via 4 partial 32-bit products (keep low 64 bits)
  let a0 = a.x & 0xFFFFu;
  let a1 = a.x >> 16u;
  let a2 = a.y & 0xFFFFu;
  let a3 = a.y >> 16u;
  let b0 = b.x & 0xFFFFu;
  let b1 = b.x >> 16u;
  let b2 = b.y & 0xFFFFu;
  let b3 = b.y >> 16u;

  // Accumulate partial products
  var acc0 = a0 * b0;
  var acc1 = a1 * b0 + a0 * b1 + (acc0 >> 16u);
  acc0 = acc0 & 0xFFFFu;
  var acc2 = a2 * b0 + a1 * b1 + a0 * b2 + (acc1 >> 16u);
  acc1 = acc1 & 0xFFFFu;
  var acc3 = a3 * b0 + a2 * b1 + a1 * b2 + a0 * b3 + (acc2 >> 16u);
  acc2 = acc2 & 0xFFFFu;
  acc3 = acc3 & 0xFFFFu;

  let lo = acc0 | (acc1 << 16u);
  let hi = acc2 | (acc3 << 16u);
  return vec2<u32>(lo, hi);
}

// ─── SplitMix64 ────────────────────────────────────────────────────────────

const SM_GOLDEN: vec2<u32> = vec2<u32>(0x7F4A7C15u, 0x9E3779B9u);
const SM_MUL1:   vec2<u32> = vec2<u32>(0x1CE4E5B9u, 0xBF58476Du);
const SM_MUL2:   vec2<u32> = vec2<u32>(0x133111EBu, 0x94D049BBu);

fn splitmix64(state: vec2<u32>) -> vec2<u32> {
  var z = u64_add(state, SM_GOLDEN);
  z = u64_mul(u64_xor(z, u64_shr(z, 30u)), SM_MUL1);
  z = u64_mul(u64_xor(z, u64_shr(z, 27u)), SM_MUL2);
  return u64_xor(z, u64_shr(z, 31u));
}

/// 2D spatial noise in [-1, 1].
fn noise_2d(x: u32, y: u32, seed_lo: u32, seed_hi: u32) -> f32 {
  let state = u64_xor(vec2<u32>(x, y), vec2<u32>(seed_lo, seed_hi));
  let h = splitmix64(state);
  return f32(h.y >> 8u) / 8388608.0 - 1.0;
}
"#;

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_different_seeds_differ() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(43);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn rng_with_offset_differs_from_bare() {
        let mut bare = Rng::new(42);
        let mut offset = Rng::with_offset(42, SEED_FILM_GRAIN);
        assert_ne!(bare.next_u64(), offset.next_u64());
    }

    #[test]
    fn rng_f32_in_range() {
        let mut rng = Rng::new(0);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn rng_f32_signed_in_range() {
        let mut rng = Rng::new(0);
        for _ in 0..1000 {
            let v = rng.next_f32_signed();
            assert!(v >= -1.0 && v <= 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn noise_2d_deterministic() {
        let a = noise_2d(10, 20, 42);
        let b = noise_2d(10, 20, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn noise_2d_different_positions_differ() {
        let a = noise_2d(10, 20, 42);
        let b = noise_2d(11, 20, 42);
        assert_ne!(a, b);
    }

    #[test]
    fn noise_2d_in_range() {
        for x in 0..100 {
            for y in 0..100 {
                let v = noise_2d(x, y, 0);
                assert!(v >= -1.0 && v <= 1.0, "out of range at ({x},{y}): {v}");
            }
        }
    }

    #[test]
    fn noise_1d_deterministic() {
        assert_eq!(noise_1d(42, 0), noise_1d(42, 0));
    }

    #[test]
    fn seed_offsets_are_unique() {
        let offsets = [
            SEED_FILM_GRAIN,
            SEED_GAUSSIAN_NOISE,
            SEED_UNIFORM_NOISE,
            SEED_SALT_PEPPER,
            SEED_POISSON_NOISE,
            SEED_GLITCH,
            SEED_DITHER,
            SEED_KMEANS,
        ];
        for i in 0..offsets.len() {
            for j in i + 1..offsets.len() {
                assert_ne!(offsets[i], offsets[j], "duplicate seed offset at {i} and {j}");
            }
        }
    }

    #[test]
    fn seeded_index_in_range() {
        for i in 0..200 {
            let idx = seeded_index(i, 42, 100);
            assert!(idx < 100, "out of range: {idx}");
        }
    }

    #[test]
    fn noise_2d_different_algo_seeds_uncorrelated() {
        // Same position, different algo offsets should produce different values
        let a = noise_2d(5, 5, 42 ^ SEED_FILM_GRAIN);
        let b = noise_2d(5, 5, 42 ^ SEED_GAUSSIAN_NOISE);
        assert_ne!(a, b);
    }

    #[test]
    fn gaussian_distribution_reasonable() {
        let mut rng = Rng::new(12345);
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let n = 10_000;
        for _ in 0..n {
            let v = rng.next_gaussian() as f64;
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        // Standard normal: mean ≈ 0, variance ≈ 1
        assert!(mean.abs() < 0.05, "mean too far from 0: {mean}");
        assert!((variance - 1.0).abs() < 0.1, "variance too far from 1: {variance}");
    }

    #[test]
    fn wgsl_snippet_has_required_functions() {
        assert!(NOISE_WGSL.contains("fn splitmix64("));
        assert!(NOISE_WGSL.contains("fn noise_2d("));
        assert!(NOISE_WGSL.contains("fn u64_mul("));
        assert!(NOISE_WGSL.contains("fn u64_add("));
        assert!(NOISE_WGSL.contains("SM_GOLDEN"));
    }
}
