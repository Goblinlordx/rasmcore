//! Camera IDT (Input Device Transform) presets.
//!
//! Each camera IDT converts from the camera's log encoding + native gamut
//! to ACES AP0 (or AP1/ACEScg) linear. Two steps:
//! 1. Linearize: apply inverse transfer function (log → linear)
//! 2. Gamut convert: 3×3 matrix from camera gamut → ACES

use crate::color_transform::{ColorTransform, ColorTransformInner, TransformKind, TransformPresetInfo};
use crate::color_space::ColorSpace;
use crate::fusion::Clut3D;
use crate::lmt::Lmt;
use crate::node::PipelineError;

/// All camera IDT preset descriptors.
pub fn camera_preset_list() -> Vec<TransformPresetInfo> {
    vec![
        TransformPresetInfo {
            name: "idt-arri-logc3", display_name: "ARRI LogC v3 (EI 800)",
            kind: TransformKind::Idt, source_space: "ARRI LogC v3", target_space: "ACEScg",
            vendor: "ARRI", description: "Alexa LogC v3 (EI 800) to ACEScg",
        },
        TransformPresetInfo {
            name: "idt-arri-logc4", display_name: "ARRI LogC v4",
            kind: TransformKind::Idt, source_space: "ARRI LogC v4", target_space: "ACEScg",
            vendor: "ARRI", description: "ARRI LogC4 (ALEXA 35) to ACEScg",
        },
        TransformPresetInfo {
            name: "idt-sony-slog3", display_name: "Sony S-Log3 / S-Gamut3",
            kind: TransformKind::Idt, source_space: "S-Log3/S-Gamut3", target_space: "ACEScg",
            vendor: "Sony", description: "Sony S-Log3 / S-Gamut3 to ACEScg",
        },
        TransformPresetInfo {
            name: "idt-red-ipp2", display_name: "RED IPP2 (Log3G10 / RWG)",
            kind: TransformKind::Idt, source_space: "REDLog3G10/RWG", target_space: "ACEScg",
            vendor: "RED", description: "RED Log3G10 / REDWideGamutRGB to ACEScg",
        },
        TransformPresetInfo {
            name: "idt-bmd-gen5", display_name: "Blackmagic Film Gen5",
            kind: TransformKind::Idt, source_space: "BMD Film Gen5", target_space: "ACEScg",
            vendor: "Blackmagic", description: "Blackmagic Film Gen5 to ACEScg",
        },
    ]
}

/// Load a camera IDT preset by name.
pub fn load_camera_preset(name: &str) -> Result<ColorTransform, PipelineError> {
    match name {
        "idt-arri-logc3" => Ok(build_camera_idt(name, arri_logc3_linearize, &ARRI_WIDE_GAMUT_TO_ACES)),
        "idt-arri-logc4" => Ok(build_camera_idt(name, arri_logc4_linearize, &ARRI_WIDE_GAMUT_4_TO_ACES)),
        "idt-sony-slog3" => Ok(build_camera_idt(name, sony_slog3_linearize, &SONY_SGAMUT3_TO_ACES)),
        "idt-red-ipp2" => Ok(build_camera_idt(name, red_log3g10_linearize, &RED_WIDE_GAMUT_TO_ACES)),
        "idt-bmd-gen5" => Ok(build_camera_idt(name, bmd_film_gen5_linearize, &BMD_WIDE_GAMUT_TO_ACES)),
        _ => Err(PipelineError::InvalidParams(format!("unknown camera IDT: {name}"))),
    }
}

fn build_camera_idt(
    name: &str,
    linearize: fn(f32) -> f32,
    gamut_matrix: &[[f32; 3]; 3],
) -> ColorTransform {
    // Build a 3D CLUT that bakes both linearize + gamut matrix.
    // Input: camera log-encoded RGB. Output: ACES AP1 linear RGB.
    // CLUT size 33 gives high accuracy for smooth log curves.
    let m = *gamut_matrix;
    let clut = Clut3D::from_fn(33, move |r, g, b| {
        // Step 1: linearize each channel
        let lr = linearize(r);
        let lg = linearize(g);
        let lb = linearize(b);
        // Step 2: gamut matrix (camera → ACES AP1)
        let or = m[0][0] * lr + m[0][1] * lg + m[0][2] * lb;
        let og = m[1][0] * lr + m[1][1] * lg + m[1][2] * lb;
        let ob = m[2][0] * lr + m[2][1] * lg + m[2][2] * lb;
        (or, og, ob)
    });

    ColorTransform {
        name: name.into(),
        kind: TransformKind::Idt,
        source_space: ColorSpace::Linear, // treated as camera-native
        target_space: ColorSpace::AcesCg,
        inner: ColorTransformInner::Lmt(Lmt::Clut3D(clut)),
    }
}

// ─── ARRI LogC v3 (EI 800) ──────────────────────────────────────────────────
// Source: ARRI LogC Curve in Linear Scene Exposure Factor, Technical Note

fn arri_logc3_linearize(x: f32) -> f32 {
    // LogC v3 EI 800 constants
    let cut = 0.010591;
    let a = 5.555556;
    let b = 0.052272;
    let c = 0.247190;
    let d = 0.385537;
    let e = 5.367655;
    let f = 0.092809;

    if x > e * cut + f {
        (10.0f32.powf((x - d) / c) - b) / a
    } else {
        (x - f) / e
    }
}

// ARRI Wide Gamut 3 → ACES AP1 (ACEScg) matrix
// Source: ACES IDT for ARRI Alexa (Academy reference)
const ARRI_WIDE_GAMUT_TO_ACES: [[f32; 3]; 3] = [
    [ 0.680206,  0.236137,  0.083658],
    [ 0.085415,  1.017471, -0.102886],
    [ 0.002057, -0.062563,  1.060506],
];

// ─── ARRI LogC v4 ────────────────────────────────────────────────────────────
// Source: ARRI LogC4 Specification (ALEXA 35)

fn arri_logc4_linearize(x: f32) -> f32 {
    let a = 2231.826309;
    let b = 64.0;
    let c = 0.074;
    let s = 7.0;
    let t = 0.01524;

    if x >= t {
        (2.0f32.powf((x - c) * s) - b) / a
    } else {
        (x - c) * s / a
    }
}

// ARRI Wide Gamut 4 → ACES AP1 (approximate, from ARRI documentation)
const ARRI_WIDE_GAMUT_4_TO_ACES: [[f32; 3]; 3] = [
    [ 0.750957,  0.144422,  0.104621],
    [ 0.000821,  1.007397, -0.008218],
    [-0.000499, -0.034855,  1.035354],
];

// ─── Sony S-Log3 / S-Gamut3 ─────────────────────────────────────────────────
// Source: Sony Technical Summary for S-Log3

fn sony_slog3_linearize(x: f32) -> f32 {
    if x >= 171.2102946929 / 1023.0 {
        10.0f32.powf((x * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01
    } else {
        (x * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0)
    }
}

// S-Gamut3 → ACES AP1 matrix (from ACES IDT for Sony cameras)
const SONY_SGAMUT3_TO_ACES: [[f32; 3]; 3] = [
    [ 0.752279,  0.143432,  0.104289],
    [-0.000156,  1.082357, -0.082201],
    [-0.000038, -0.067961,  1.067999],
];

// ─── RED Log3G10 / REDWideGamutRGB ──────────────────────────────────────────
// Source: RED Technical White Paper on IPP2

fn red_log3g10_linearize(x: f32) -> f32 {
    let a = 0.224282;
    let b = 155.975327;
    let c = 0.01;
    let g = 15.1927;

    if x < 0.0 {
        (x / g) - c
    } else {
        (10.0f32.powf(x / a) - 1.0) / b - c
    }
}

// REDWideGamutRGB → ACES AP1 matrix (from ACES IDT for RED cameras)
const RED_WIDE_GAMUT_TO_ACES: [[f32; 3]; 3] = [
    [ 0.785043,  0.083844,  0.131118],
    [ 0.023172,  1.087892, -0.111055],
    [-0.073769, -0.314639,  1.388408],
];

// ─── Blackmagic Film Gen5 ───────────────────────────────────────────────────
// Source: Blackmagic Design documentation

fn bmd_film_gen5_linearize(x: f32) -> f32 {
    let a: f32 = 8.283064;
    let b: f32 = 0.09246575;
    let c: f32 = 0.005494;
    let d: f32 = 0.5;
    let e: f32 = 10.44426;
    let lin_cut: f32 = 0.005;

    let t = d + (1.0 / a.ln()) * (x * a + b).max(1e-10).ln();
    if t > lin_cut { t } else { (x - c) / e }
}

// Blackmagic Wide Gamut → ACES AP1 matrix (approximate)
const BMD_WIDE_GAMUT_TO_ACES: [[f32; 3]; 3] = [
    [ 0.638008,  0.214704,  0.147288],
    [ 0.058137,  0.967925, -0.026062],
    [-0.002070, -0.062309,  1.064379],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arri_logc3_midgray() {
        // LogC v3 mid-gray (18% reflectance) encodes to ~0.391
        let encoded = 0.391;
        let linear = arri_logc3_linearize(encoded);
        assert!((linear - 0.18).abs() < 0.02, "LogC v3 mid-gray: {linear}");
    }

    #[test]
    fn sony_slog3_midgray() {
        // S-Log3 mid-gray (18%) encodes to ~420/1023 ≈ 0.4105
        let encoded = 420.0 / 1023.0;
        let linear = sony_slog3_linearize(encoded);
        assert!((linear - 0.18).abs() < 0.02, "S-Log3 mid-gray: {linear}");
    }

    #[test]
    fn camera_preset_list_has_entries() {
        let list = camera_preset_list();
        assert!(list.len() >= 5);
        assert!(list.iter().any(|p| p.name == "idt-arri-logc3"));
        assert!(list.iter().any(|p| p.name == "idt-sony-slog3"));
        assert!(list.iter().any(|p| p.name == "idt-red-ipp2"));
    }

    #[test]
    fn load_arri_logc3() {
        let transform = load_camera_preset("idt-arri-logc3").unwrap();
        assert_eq!(transform.kind, TransformKind::Idt);
        // Apply to mid-gray encoded pixel
        let mut pixels = vec![0.391, 0.391, 0.391, 1.0];
        transform.apply(&mut pixels);
        // Should be in ACES AP1 linear space, near 0.18
        assert!(pixels[0] > 0.05 && pixels[0] < 0.5, "ARRI mid-gray in ACEScg: {}", pixels[0]);
    }

    #[test]
    fn load_unknown_preset_errors() {
        assert!(load_camera_preset("idt-nonexistent").is_err());
    }
}
