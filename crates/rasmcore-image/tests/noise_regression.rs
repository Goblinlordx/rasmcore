//! Noise regression tests — scalar snapshots for SIMD validation.
//!
//! These fixtures capture the EXACT u8 output of the scalar noise implementation
//! at known (seed, scale, octaves) parameters. Any change to the noise functions
//! (including f32 conversion or SIMD) MUST produce identical u8 output, or the
//! fixtures must be deliberately updated with justification.
//!
//! Fixture generation: /tmp/gen_noise_fixtures.rs (or regenerate from current code)
//! Fixture location: tests/fixtures/noise/

use rasmcore_image::domain::filters;

fn load_fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/noise/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {path}: {e}"))
}

#[test]
fn perlin_64x64_seed42_matches_fixture() {
    let fixture = load_fixture("perlin_64x64_seed42_s005_o4.raw");
    let result = filters::perlin_noise(64, 64, 42, 0.05, 4);
    assert_eq!(result.len(), fixture.len());
    assert_eq!(
        result, fixture,
        "Perlin 64x64 seed=42 output changed — SIMD must match scalar snapshot"
    );
}

#[test]
fn perlin_128x128_seed1_matches_fixture() {
    let fixture = load_fixture("perlin_128x128_seed1_s002_o6.raw");
    let result = filters::perlin_noise(128, 128, 1, 0.02, 6);
    assert_eq!(result.len(), fixture.len());
    assert_eq!(
        result, fixture,
        "Perlin 128x128 seed=1 output changed — SIMD must match scalar snapshot"
    );
}

#[test]
fn simplex_64x64_seed42_matches_fixture() {
    let fixture = load_fixture("simplex_64x64_seed42_s005_o4.raw");
    let result = filters::simplex_noise(64, 64, 42, 0.05, 4);
    assert_eq!(result.len(), fixture.len());
    assert_eq!(
        result, fixture,
        "Simplex 64x64 seed=42 output changed — SIMD must match scalar snapshot"
    );
}

#[test]
fn simplex_128x128_seed1_matches_fixture() {
    let fixture = load_fixture("simplex_128x128_seed1_s002_o6.raw");
    let result = filters::simplex_noise(128, 128, 1, 0.02, 6);
    assert_eq!(result.len(), fixture.len());
    assert_eq!(
        result, fixture,
        "Simplex 128x128 seed=1 output changed — SIMD must match scalar snapshot"
    );
}
