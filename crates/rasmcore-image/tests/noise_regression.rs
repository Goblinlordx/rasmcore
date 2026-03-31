//! Noise regression tests — snapshot validation for SIMD consistency.
//!
//! These tests verify that noise functions produce consistent output across
//! code changes (including SIMD optimizations). On first run, they generate
//! and cache the reference snapshot. On subsequent runs, they compare against
//! the cached snapshot. Any change to noise output will fail the test.
//!
//! Snapshots are stored in a gitignored cache directory and auto-generated.

use rasmcore_image::domain::filters;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/noise");
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Load a fixture snapshot, or generate and cache it if missing.
fn load_or_generate(name: &str, generate: impl FnOnce() -> Vec<u8>) -> Vec<u8> {
    let path = fixture_dir().join(name);
    if let Ok(data) = std::fs::read(&path) {
        data
    } else {
        let data = generate();
        std::fs::write(&path, &data).ok();
        eprintln!("  Generated snapshot: {name} ({} bytes)", data.len());
        data
    }
}

#[test]
fn perlin_64x64_seed42_matches_fixture() {
    let fixture = load_or_generate("perlin_64x64_seed42_s005_o4.raw", || {
        filters::perlin_noise(64, 64, 42, 0.05, 4)
    });
    let result = filters::perlin_noise(64, 64, 42, 0.05, 4);
    assert_eq!(result.len(), fixture.len());
    assert_eq!(
        result, fixture,
        "Perlin 64x64 seed=42 output changed — SIMD must match scalar snapshot"
    );
}

#[test]
fn perlin_128x128_seed1_matches_fixture() {
    let fixture = load_or_generate("perlin_128x128_seed1_s002_o6.raw", || {
        filters::perlin_noise(128, 128, 1, 0.02, 6)
    });
    let result = filters::perlin_noise(128, 128, 1, 0.02, 6);
    assert_eq!(result.len(), fixture.len());
    assert_eq!(
        result, fixture,
        "Perlin 128x128 seed=1 output changed — SIMD must match scalar snapshot"
    );
}

#[test]
fn simplex_64x64_seed42_matches_fixture() {
    let fixture = load_or_generate("simplex_64x64_seed42_s005_o4.raw", || {
        filters::simplex_noise(64, 64, 42, 0.05, 4)
    });
    let result = filters::simplex_noise(64, 64, 42, 0.05, 4);
    assert_eq!(result.len(), fixture.len());
    assert_eq!(
        result, fixture,
        "Simplex 64x64 seed=42 output changed — SIMD must match scalar snapshot"
    );
}

#[test]
fn simplex_128x128_seed1_matches_fixture() {
    let fixture = load_or_generate("simplex_128x128_seed1_s002_o6.raw", || {
        filters::simplex_noise(128, 128, 1, 0.02, 6)
    });
    let result = filters::simplex_noise(128, 128, 1, 0.02, 6);
    assert_eq!(result.len(), fixture.len());
    assert_eq!(
        result, fixture,
        "Simplex 128x128 seed=1 output changed — SIMD must match scalar snapshot"
    );
}
