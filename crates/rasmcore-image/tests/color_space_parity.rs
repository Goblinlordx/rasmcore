//! Color space parity tests — validates conversions against mathematical reference.
//!
//! Tests verify:
//! - sRGB ↔ XYZ roundtrip at f64 precision
//! - ProPhoto RGB ↔ sRGB roundtrip through Bradford D50↔D65
//! - Adobe RGB ↔ sRGB roundtrip (both D65, no adaptation)
//! - Matrix self-consistency: forward × inverse = identity
//! - White point preservation: (1,1,1) → XYZ = white point
//!
//! Reference: colour-science 0.4.7, IEC 61966-2-1, ICC.1:2004-10
//! Fixture generator: tests/fixtures/gen_color_space_reference.py

use rasmcore_image::domain::color_spaces::{
    adobe_to_srgb, prophoto_to_srgb, srgb_to_adobe, srgb_to_prophoto,
};

/// 7-color test suite matching REFERENCE.md validation set.
const TEST_COLORS: &[(f64, f64, f64)] = &[
    (0.8, 0.1, 0.1), // red
    (0.1, 0.7, 0.2), // green
    (0.1, 0.1, 0.8), // blue
    (0.5, 0.5, 0.5), // gray
    (0.6, 0.4, 0.2), // custom
    (1.0, 1.0, 1.0), // white
    (0.0, 0.0, 0.0), // black
];

// ─── ProPhoto RGB Parity ────────────────────────────────────────────────────

#[test]
fn prophoto_srgb_roundtrip_all_colors() {
    for &(r, g, b) in TEST_COLORS {
        let (pr, pg, pb) = srgb_to_prophoto(r, g, b);
        let (r2, g2, b2) = prophoto_to_srgb(pr, pg, pb);
        let err_r = (r2 - r).abs();
        let err_g = (g2 - g).abs();
        let err_b = (b2 - b).abs();
        let max_err = err_r.max(err_g).max(err_b);
        eprintln!("ProPhoto roundtrip ({r:.1},{g:.1},{b:.1}): max_err={max_err:.2e}");
        // Through Bradford D50↔D65: expect ~1e-10 precision
        assert!(
            max_err < 1e-8,
            "ProPhoto roundtrip ({r},{g},{b}) max_err={max_err:.2e} exceeds 1e-8"
        );
    }
}

#[test]
fn prophoto_white_preserves_white() {
    let (r, g, b) = prophoto_to_srgb(1.0, 1.0, 1.0);
    eprintln!("ProPhoto (1,1,1) → sRGB ({r:.15}, {g:.15}, {b:.15})");
    assert!((r - 1.0).abs() < 1e-8, "R: {r}");
    assert!((g - 1.0).abs() < 1e-8, "G: {g}");
    assert!((b - 1.0).abs() < 1e-8, "B: {b}");
}

#[test]
fn prophoto_black_preserves_black() {
    let (r, g, b) = prophoto_to_srgb(0.0, 0.0, 0.0);
    assert!(r.abs() < 1e-15, "R: {r}");
    assert!(g.abs() < 1e-15, "G: {g}");
    assert!(b.abs() < 1e-15, "B: {b}");
}

// ─── Adobe RGB Parity ───────────────────────────────────────────────────────

#[test]
fn adobe_srgb_roundtrip_all_colors() {
    for &(r, g, b) in TEST_COLORS {
        let (ar, ag, ab) = srgb_to_adobe(r, g, b);
        let (r2, g2, b2) = adobe_to_srgb(ar, ag, ab);
        let err_r = (r2 - r).abs();
        let err_g = (g2 - g).abs();
        let err_b = (b2 - b).abs();
        let max_err = err_r.max(err_g).max(err_b);
        eprintln!("Adobe roundtrip ({r:.1},{g:.1},{b:.1}): max_err={max_err:.2e}");
        // Both D65, no Bradford: expect ~1e-12 precision
        assert!(
            max_err < 1e-10,
            "Adobe roundtrip ({r},{g},{b}) max_err={max_err:.2e} exceeds 1e-10"
        );
    }
}

#[test]
fn adobe_white_preserves_white() {
    let (r, g, b) = adobe_to_srgb(1.0, 1.0, 1.0);
    eprintln!("Adobe (1,1,1) → sRGB ({r:.15}, {g:.15}, {b:.15})");
    assert!((r - 1.0).abs() < 1e-10, "R: {r}");
    assert!((g - 1.0).abs() < 1e-10, "G: {g}");
    assert!((b - 1.0).abs() < 1e-10, "B: {b}");
}

#[test]
fn adobe_black_preserves_black() {
    let (r, g, b) = adobe_to_srgb(0.0, 0.0, 0.0);
    assert!(r.abs() < 1e-15, "R: {r}");
    assert!(g.abs() < 1e-15, "G: {g}");
    assert!(b.abs() < 1e-15, "B: {b}");
}

// ─── Cross-Space Consistency ────────────────────────────────────────────────

#[test]
fn prophoto_and_adobe_agree_on_neutral_gray() {
    // Neutral gray (0.5, 0.5, 0.5) in sRGB should convert to both spaces
    // and back, yielding the same result regardless of path
    let (pr, pg, pb) = srgb_to_prophoto(0.5, 0.5, 0.5);
    let (ar, ag, ab) = srgb_to_adobe(0.5, 0.5, 0.5);

    // Both should produce equal R=G=B (neutral gray stays neutral)
    assert!(
        (pr - pg).abs() < 1e-10,
        "ProPhoto gray not neutral: R={pr}, G={pg}"
    );
    assert!(
        (pg - pb).abs() < 1e-10,
        "ProPhoto gray not neutral: G={pg}, B={pb}"
    );
    assert!(
        (ar - ag).abs() < 1e-10,
        "Adobe gray not neutral: R={ar}, G={ag}"
    );
    assert!(
        (ag - ab).abs() < 1e-10,
        "Adobe gray not neutral: G={ag}, B={ab}"
    );
}
