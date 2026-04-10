//! Filter reference parity tests — compare our optimized implementations
//! against the rasmcore-reference crate's textbook implementations.
//!
//! These tests use procedurally generated inputs (no committed images).
//! The reference implementations are independently validated against
//! production tools (see rasmcore-reference/VALIDATION.md).

use rasmcore_pipeline_v2::ops::Filter;
use rasmcore_reference as refimpl;

const W: u32 = 32;
const H: u32 = 32;
const POINT_TOL: f32 = 1e-6;
const COLOR_TOL: f32 = 1e-4;

// ─── Point Ops ──────────────────────────────────────────────────────────────

#[test]
fn ref_brightness() {
    let input = refimpl::gradient(W, H);
    for amount in [-0.5, -0.1, 0.0, 0.1, 0.3, 0.8] {
        let expected = refimpl::point_ops::brightness(&input, W, H, amount);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Brightness { amount }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(
            &format!("brightness({amount})"),
            &actual,
            &expected,
            POINT_TOL,
        );
    }
}

#[test]
fn ref_contrast() {
    let input = refimpl::gradient(W, H);
    for amount in [-0.5, 0.0, 0.3, 0.8] {
        let expected = refimpl::point_ops::contrast(&input, W, H, amount);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Contrast { amount }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(
            &format!("contrast({amount})"),
            &actual,
            &expected,
            POINT_TOL,
        );
    }
}

#[test]
fn ref_gamma() {
    let input = refimpl::gradient(W, H);
    for gamma in [0.5, 1.0, 1.5, 2.2, 3.0] {
        let expected = refimpl::point_ops::gamma(&input, W, H, gamma);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Gamma { gamma }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(&format!("gamma({gamma})"), &actual, &expected, POINT_TOL);
    }
}

#[test]
fn ref_exposure() {
    let input = refimpl::gradient(W, H);
    for ev in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let expected = refimpl::point_ops::exposure(&input, W, H, ev);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Exposure { ev }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(&format!("exposure({ev})"), &actual, &expected, POINT_TOL);
    }
}

#[test]
fn ref_invert() {
    let input = refimpl::gradient(W, H);
    let expected = refimpl::point_ops::invert(&input, W, H);
    let actual = rasmcore_pipeline_v2::filters::adjustment::Invert
        .compute(&input, W, H)
        .unwrap();
    refimpl::assert_parity("invert", &actual, &expected, POINT_TOL);
}

#[test]
fn ref_levels() {
    let input = refimpl::gradient(W, H);
    for (black, white, gamma) in [
        (0.0, 1.0, 1.0),
        (0.1, 0.9, 1.0),
        (0.0, 1.0, 2.2),
        (0.2, 0.8, 0.5),
    ] {
        let expected = refimpl::point_ops::levels(&input, W, H, black, white, gamma);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Levels {
            black,
            white,
            gamma,
        }
        .compute(&input, W, H)
        .unwrap();
        refimpl::assert_parity(
            &format!("levels({black},{white},{gamma})"),
            &actual,
            &expected,
            POINT_TOL,
        );
    }
}

#[test]
fn ref_posterize() {
    let input = refimpl::gradient(W, H);
    for levels in [2u8, 4, 8, 16, 64] {
        let expected = refimpl::point_ops::posterize(&input, W, H, levels as u32);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Posterize { levels }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(
            &format!("posterize({levels})"),
            &actual,
            &expected,
            POINT_TOL,
        );
    }
}

#[test]
fn ref_solarize() {
    let input = refimpl::gradient(W, H);
    for threshold in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let expected = refimpl::point_ops::solarize(&input, W, H, threshold);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Solarize { threshold }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(
            &format!("solarize({threshold})"),
            &actual,
            &expected,
            POINT_TOL,
        );
    }
}

#[test]
fn ref_sigmoidal_contrast() {
    let input = refimpl::gradient(W, H);
    for (strength, midpoint) in [(3.0, 0.5), (5.0, 0.3), (10.0, 0.5)] {
        let expected = refimpl::point_ops::sigmoidal_contrast(&input, W, H, strength, midpoint);
        let actual = rasmcore_pipeline_v2::filters::adjustment::SigmoidalContrast {
            strength,
            midpoint,
            sharpen: true,
        }
        .compute(&input, W, H)
        .unwrap();
        refimpl::assert_parity(
            &format!("sigmoidal({strength},{midpoint})"),
            &actual,
            &expected,
            1e-5,
        );
    }
}

#[test]
fn ref_dodge() {
    let input = refimpl::gradient(W, H);
    for amount in [0.0, 0.3, 0.5, 0.8] {
        let expected = refimpl::point_ops::dodge(&input, W, H, amount);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Dodge { amount }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(&format!("dodge({amount})"), &actual, &expected, POINT_TOL);
    }
}

#[test]
fn ref_burn() {
    let input = refimpl::gradient(W, H);
    for amount in [0.3, 0.5, 1.0, 1.5] {
        let expected = refimpl::point_ops::burn(&input, W, H, amount);
        let actual = rasmcore_pipeline_v2::filters::adjustment::Burn { amount }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(&format!("burn({amount})"), &actual, &expected, POINT_TOL);
    }
}

// ─── Color Ops ──────────────────────────────────────────────────────────────

#[test]
fn ref_sepia() {
    let input = refimpl::gradient(W, H);
    for intensity in [0.0, 0.5, 1.0] {
        let expected = refimpl::color_ops::sepia(&input, W, H, intensity);
        let actual = rasmcore_pipeline_v2::filters::color::Sepia { intensity }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(
            &format!("sepia({intensity})"),
            &actual,
            &expected,
            COLOR_TOL,
        );
    }
}

#[test]
fn ref_saturate() {
    let input = refimpl::gradient(W, H);
    for factor in [0.0, 0.5, 1.0, 1.5] {
        let expected = refimpl::color_ops::saturate(&input, W, H, factor);
        let actual = rasmcore_pipeline_v2::filters::color::Saturate { factor }
            .compute(&input, W, H)
            .unwrap();
        refimpl::assert_parity(
            &format!("saturate({factor})"),
            &actual,
            &expected,
            COLOR_TOL,
        );
    }
}

#[test]
fn ref_channel_mixer() {
    let input = refimpl::gradient(W, H);
    // Identity
    let id_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let expected = refimpl::color_ops::channel_mixer(&input, W, H, &id_matrix);
    let actual = rasmcore_pipeline_v2::filters::color::ChannelMixer { matrix: id_matrix }
        .compute(&input, W, H)
        .unwrap();
    refimpl::assert_parity("channel_mixer(identity)", &actual, &expected, POINT_TOL);

    // Swap R/B
    let swap_matrix = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let expected = refimpl::color_ops::channel_mixer(&input, W, H, &swap_matrix);
    let actual = rasmcore_pipeline_v2::filters::color::ChannelMixer {
        matrix: swap_matrix,
    }
    .compute(&input, W, H)
    .unwrap();
    refimpl::assert_parity("channel_mixer(swap_rb)", &actual, &expected, POINT_TOL);
}

// ─── Multi-input validation (noise, solid) ──────────────────────────────────

#[test]
fn ref_brightness_on_noise() {
    let input = refimpl::noise(W, H, 42);
    let expected = refimpl::point_ops::brightness(&input, W, H, 0.2);
    let actual = rasmcore_pipeline_v2::filters::adjustment::Brightness { amount: 0.2 }
        .compute(&input, W, H)
        .unwrap();
    refimpl::assert_parity("brightness_noise", &actual, &expected, POINT_TOL);
}

#[test]
fn ref_invert_on_solid() {
    let input = refimpl::solid(W, H, [0.3, 0.6, 0.9, 1.0]);
    let expected = refimpl::point_ops::invert(&input, W, H);
    let actual = rasmcore_pipeline_v2::filters::adjustment::Invert
        .compute(&input, W, H)
        .unwrap();
    refimpl::assert_parity("invert_solid", &actual, &expected, POINT_TOL);
}
