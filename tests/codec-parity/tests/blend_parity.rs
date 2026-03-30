//! Pixel-exact parity tests for all 19 blend modes.
//!
//! Compares our Rust `blend()` implementation against:
//! - libvips 8.18 (modes it supports: multiply, screen, overlay, darken, lighten,
//!   colour-dodge, colour-burn, hard-light, soft-light, difference, exclusion)
//! - ImageMagick 7 (all modes including extended: vivid-light, linear-dodge,
//!   linear-burn, linear-light, pin-light, hard-mix, subtract, divide)
//!
//! Tolerance: max +/-1 per channel to account for FP rounding.
//!
//! Known reference divergence:
//! - SoftLight: IM 7 Q16-HDRI uses a different formula than the W3C spec.
//!   We follow the W3C/vips formula and only validate against vips for this mode.

use rasmcore_image::domain::filters::{BlendMode, blend};
use rasmcore_image::domain::types::{ColorSpace, ImageInfo, PixelFormat};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Test image generation ────────────────────────────────────────────

const W: u32 = 8;
const H: u32 = 8;

/// Build a gradient image: each of the 64 pixels has a different R/G/B value.
/// fg image ramps R up, G down, B in a triangle pattern.
/// bg image ramps R down, G up, B constant-ish.
fn make_test_images() -> (Vec<u8>, Vec<u8>) {
    let n = (W * H) as usize;
    let mut fg = Vec::with_capacity(n * 3);
    let mut bg = Vec::with_capacity(n * 3);

    for i in 0..n {
        let t = i as f32 / (n - 1) as f32; // 0..1

        // FG: diverse gradient
        let fr = (t * 255.0 + 0.5) as u8;
        let fg_val = ((1.0 - t) * 255.0 + 0.5) as u8;
        let fb = ((0.5 - (t - 0.5).abs()) * 2.0 * 255.0 + 0.5) as u8;
        fg.push(fr);
        fg.push(fg_val);
        fg.push(fb);

        // BG: complementary gradient
        let br = ((1.0 - t) * 200.0 + 0.5) as u8;
        let bg_val = (t * 200.0 + 55.5) as u8;
        let bb = 128;
        bg.push(br);
        bg.push(bg_val);
        bg.push(bb);
    }

    (fg, bg)
}

/// Additional edge-case solid-color pairs that exercise boundary values.
fn edge_case_pairs() -> Vec<([u8; 3], [u8; 3], &'static str)> {
    vec![
        ([0, 0, 0], [0, 0, 0], "black_on_black"),
        ([255, 255, 255], [255, 255, 255], "white_on_white"),
        ([0, 0, 0], [255, 255, 255], "black_on_white"),
        ([255, 255, 255], [0, 0, 0], "white_on_black"),
        ([128, 128, 128], [128, 128, 128], "mid_on_mid"),
        ([200, 50, 1], [50, 200, 254], "mixed_extremes"),
        ([1, 1, 1], [254, 254, 254], "near_bounds"),
    ]
}

/// Encode raw RGB pixels as a PPM binary (P6) file.
fn encode_ppm(w: u32, h: u32, pixels: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    write!(buf, "P6\n{w} {h}\n255\n").unwrap();
    buf.extend_from_slice(pixels);
    buf
}

/// Parse a PPM (P6) file back to (width, height, rgb_pixels).
/// Handles optional comment lines (vips adds one).
fn parse_ppm(data: &[u8]) -> (u32, u32, Vec<u8>) {
    let mut pos = 0;

    // Skip magic "P6\n"
    assert!(data.starts_with(b"P6"), "not a P6 PPM file");
    pos = data.iter().position(|&b| b == b'\n').unwrap() + 1;

    // Skip comment lines
    while pos < data.len() && data[pos] == b'#' {
        pos = data[pos..].iter().position(|&b| b == b'\n').unwrap() + pos + 1;
    }

    // Read dimensions
    let dim_end = data[pos..].iter().position(|&b| b == b'\n').unwrap() + pos;
    let dim_str = std::str::from_utf8(&data[pos..dim_end]).unwrap();
    let mut parts = dim_str.split_whitespace();
    let w: u32 = parts.next().unwrap().parse().unwrap();
    let h: u32 = parts.next().unwrap().parse().unwrap();
    pos = dim_end + 1;

    // Read maxval
    let max_end = data[pos..].iter().position(|&b| b == b'\n').unwrap() + pos;
    pos = max_end + 1;

    let pixels = data[pos..].to_vec();
    assert_eq!(
        pixels.len(),
        (w * h * 3) as usize,
        "PPM pixel data length mismatch: got {} expected {} ({}x{}x3)",
        pixels.len(),
        w * h * 3,
        w,
        h
    );
    (w, h, pixels)
}

fn make_image_info(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    }
}

// ── Reference tool runners ───────────────────────────────────────────

/// Create a unique temporary directory for test artifacts.
fn make_temp_dir(prefix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("blend_parity_{prefix}_{pid}_{id}"));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Run vips composite and return the output RGB pixels.
///
/// Layer ordering: vips `composite "A B" out mode` treats A as the source layer
/// and B as the backdrop.  For asymmetric blend formulas (overlay, soft-light, etc.)
/// the W3C spec condition checks the *backdrop* value, so we pass bg as A and fg
/// as B, which empirically produces results matching our `blend_channel(fg, bg)`.
fn run_vips(fg_ppm: &[u8], bg_ppm: &[u8], mode_num: u32) -> Vec<u8> {
    let dir = make_temp_dir("vips");
    let fg_path = dir.join("fg.ppm");
    let bg_path = dir.join("bg.ppm");
    let out_path = dir.join("out.ppm");

    std::fs::write(&fg_path, fg_ppm).unwrap();
    std::fs::write(&bg_path, bg_ppm).unwrap();

    let file_list = format!(
        "{} {}",
        bg_path.to_str().unwrap(),
        fg_path.to_str().unwrap()
    );

    let output = Command::new("vips")
        .args([
            "composite",
            &file_list,
            out_path.to_str().unwrap(),
            &mode_num.to_string(),
            "--compositing-space",
            "srgb",
        ])
        .output()
        .expect("failed to run vips");

    if !output.status.success() {
        panic!(
            "vips failed (mode {}): {}",
            mode_num,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let ppm_data = std::fs::read(&out_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (_, _, pixels) = parse_ppm(&ppm_data);
    pixels
}

/// Run ImageMagick and return the output RGB pixels.
fn run_magick(fg_ppm: &[u8], bg_ppm: &[u8], mode_name: &str) -> Vec<u8> {
    let dir = make_temp_dir("magick");
    let fg_path = dir.join("fg.ppm");
    let bg_path = dir.join("bg.ppm");
    let out_path = dir.join("out.ppm");

    std::fs::write(&fg_path, fg_ppm).unwrap();
    std::fs::write(&bg_path, bg_ppm).unwrap();

    // magick bg.ppm fg.ppm -compose Mode -composite out.ppm
    let output = Command::new("magick")
        .args([
            bg_path.to_str().unwrap(),
            fg_path.to_str().unwrap(),
            "-compose",
            mode_name,
            "-composite",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run magick");

    if !output.status.success() {
        panic!(
            "magick failed (mode {}): {}",
            mode_name,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let ppm_data = std::fs::read(&out_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (_, _, pixels) = parse_ppm(&ppm_data);
    pixels
}

// ── Comparison ───────────────────────────────────────────────────────

/// Compare two pixel buffers. Returns (max_diff, num_differing_channels).
fn compare_pixels(ours: &[u8], reference: &[u8]) -> (u8, usize) {
    assert_eq!(ours.len(), reference.len(), "pixel buffer length mismatch");
    let mut max_diff: u8 = 0;
    let mut num_diff = 0;
    for (i, (&a, &b)) in ours.iter().zip(reference.iter()).enumerate() {
        let diff = (a as i16 - b as i16).unsigned_abs() as u8;
        if diff > 0 {
            num_diff += 1;
        }
        if diff > max_diff {
            max_diff = diff;
            if diff > 1 {
                let px = i / 3;
                let ch = i % 3;
                eprintln!("  MISMATCH at pixel {px} channel {ch}: ours={a} ref={b} diff={diff}");
            }
        }
    }
    (max_diff, num_diff)
}

// ── Mode definitions ─────────────────────────────────────────────────

struct ModeSpec {
    blend_mode: BlendMode,
    name: &'static str,
    /// vips mode number (None if vips doesn't support it)
    vips_num: Option<u32>,
    /// ImageMagick compose operator name
    magick_name: &'static str,
    /// Skip ImageMagick cross-validation (IM uses different formula)
    skip_magick_cross: bool,
}

fn all_modes() -> Vec<ModeSpec> {
    vec![
        ModeSpec {
            blend_mode: BlendMode::Multiply,
            name: "Multiply",
            vips_num: Some(14),
            magick_name: "Multiply",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::Screen,
            name: "Screen",
            vips_num: Some(15),
            magick_name: "Screen",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::Overlay,
            name: "Overlay",
            vips_num: Some(16),
            magick_name: "Overlay",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::Darken,
            name: "Darken",
            vips_num: Some(17),
            magick_name: "Darken",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::Lighten,
            name: "Lighten",
            vips_num: Some(18),
            magick_name: "Lighten",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::ColorDodge,
            name: "ColorDodge",
            vips_num: Some(19),
            magick_name: "ColorDodge",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::ColorBurn,
            name: "ColorBurn",
            vips_num: Some(20),
            magick_name: "ColorBurn",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::HardLight,
            name: "HardLight",
            vips_num: Some(21),
            magick_name: "HardLight",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::SoftLight,
            name: "SoftLight",
            vips_num: Some(22),
            magick_name: "SoftLight",
            // IM 7 Q16-HDRI SoftLight uses a different formula than the W3C spec.
            // vips matches W3C. We validate against vips only.
            skip_magick_cross: true,
        },
        ModeSpec {
            blend_mode: BlendMode::Difference,
            name: "Difference",
            vips_num: Some(23),
            magick_name: "Difference",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::Exclusion,
            name: "Exclusion",
            vips_num: Some(24),
            magick_name: "Exclusion",
            skip_magick_cross: false,
        },
        // Extended modes (ImageMagick only)
        ModeSpec {
            blend_mode: BlendMode::VividLight,
            name: "VividLight",
            vips_num: None,
            magick_name: "VividLight",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::LinearDodge,
            name: "LinearDodge",
            vips_num: None,
            magick_name: "LinearDodge",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::LinearBurn,
            name: "LinearBurn",
            vips_num: None,
            magick_name: "LinearBurn",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::LinearLight,
            name: "LinearLight",
            vips_num: None,
            magick_name: "LinearLight",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::PinLight,
            name: "PinLight",
            vips_num: None,
            magick_name: "PinLight",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::HardMix,
            name: "HardMix",
            vips_num: None,
            magick_name: "HardMix",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::Subtract,
            name: "Subtract",
            vips_num: None,
            // IM MinusSrc = Dest - Src = bg - fg, matching our Subtract semantics
            magick_name: "MinusSrc",
            skip_magick_cross: false,
        },
        ModeSpec {
            blend_mode: BlendMode::Divide,
            name: "Divide",
            vips_num: None,
            // IM DivideDst = Dest / Src = bg / fg, matching our Divide semantics
            magick_name: "DivideDst",
            skip_magick_cross: false,
        },
    ]
}

// ── Tests ────────────────────────────────────────────────────────────

const TOLERANCE: u8 = 1;

/// Run parity for a single mode against gradient images and edge cases.
fn test_mode(spec: &ModeSpec) {
    let (fg_raw, bg_raw) = make_test_images();
    let fg_ppm = encode_ppm(W, H, &fg_raw);
    let bg_ppm = encode_ppm(W, H, &bg_raw);

    let info = make_image_info(W, H);
    let ours = blend(&fg_raw, &info, &bg_raw, &info, spec.blend_mode).unwrap();

    // Test against vips if supported, otherwise ImageMagick
    let (ref_pixels, ref_tool) = if let Some(vips_num) = spec.vips_num {
        (run_vips(&fg_ppm, &bg_ppm, vips_num), "vips")
    } else {
        (run_magick(&fg_ppm, &bg_ppm, spec.magick_name), "magick")
    };

    let (max_diff, num_diff) = compare_pixels(&ours, &ref_pixels);
    eprintln!(
        "[GRADIENT] {:15} vs {ref_tool:6}: max_diff={max_diff}, \
         differing_channels={num_diff}/{}",
        spec.name,
        ours.len()
    );
    assert!(
        max_diff <= TOLERANCE,
        "{}: gradient test FAILED against {ref_tool} - max diff {max_diff} \
         exceeds tolerance {TOLERANCE}",
        spec.name
    );

    // Cross-validate against ImageMagick if we used vips above
    if spec.vips_num.is_some() && !spec.skip_magick_cross {
        let magick_ref = run_magick(&fg_ppm, &bg_ppm, spec.magick_name);
        let (max_diff_m, num_diff_m) = compare_pixels(&ours, &magick_ref);

        eprintln!(
            "[GRADIENT] {:15} vs magick: max_diff={max_diff_m}, \
             differing_channels={num_diff_m}/{}",
            spec.name,
            ours.len()
        );

        assert!(
            max_diff_m <= TOLERANCE,
            "{}: gradient test FAILED against magick - max diff {max_diff_m} \
             exceeds tolerance {TOLERANCE}",
            spec.name
        );
    }

    // Edge case solid-color tests
    for (fg_color, bg_color, pair_name) in edge_case_pairs() {
        let fg_solid: Vec<u8> = (0..W * H).flat_map(|_| fg_color.iter().copied()).collect();
        let bg_solid: Vec<u8> = (0..W * H).flat_map(|_| bg_color.iter().copied()).collect();
        let fg_solid_ppm = encode_ppm(W, H, &fg_solid);
        let bg_solid_ppm = encode_ppm(W, H, &bg_solid);

        let ours_solid = blend(&fg_solid, &info, &bg_solid, &info, spec.blend_mode).unwrap();

        let ref_solid = if let Some(vips_num) = spec.vips_num {
            run_vips(&fg_solid_ppm, &bg_solid_ppm, vips_num)
        } else {
            run_magick(&fg_solid_ppm, &bg_solid_ppm, spec.magick_name)
        };

        let (max_diff_s, _) = compare_pixels(&ours_solid, &ref_solid);
        assert!(
            max_diff_s <= TOLERANCE,
            "{}: edge case '{pair_name}' FAILED - max diff {max_diff_s} \
             exceeds tolerance {TOLERANCE}",
            spec.name
        );
    }
}

// Individual test functions for each mode so failures are isolated

#[test]
fn blend_parity_multiply() {
    test_mode(&all_modes()[0]);
}

#[test]
fn blend_parity_screen() {
    test_mode(&all_modes()[1]);
}

#[test]
fn blend_parity_overlay() {
    test_mode(&all_modes()[2]);
}

#[test]
fn blend_parity_darken() {
    test_mode(&all_modes()[3]);
}

#[test]
fn blend_parity_lighten() {
    test_mode(&all_modes()[4]);
}

#[test]
fn blend_parity_color_dodge() {
    test_mode(&all_modes()[5]);
}

#[test]
fn blend_parity_color_burn() {
    test_mode(&all_modes()[6]);
}

#[test]
fn blend_parity_hard_light() {
    test_mode(&all_modes()[7]);
}

#[test]
fn blend_parity_soft_light() {
    test_mode(&all_modes()[8]);
}

#[test]
fn blend_parity_difference() {
    test_mode(&all_modes()[9]);
}

#[test]
fn blend_parity_exclusion() {
    test_mode(&all_modes()[10]);
}

#[test]
fn blend_parity_vivid_light() {
    test_mode(&all_modes()[11]);
}

#[test]
fn blend_parity_linear_dodge() {
    test_mode(&all_modes()[12]);
}

#[test]
fn blend_parity_linear_burn() {
    test_mode(&all_modes()[13]);
}

#[test]
fn blend_parity_linear_light() {
    test_mode(&all_modes()[14]);
}

#[test]
fn blend_parity_pin_light() {
    test_mode(&all_modes()[15]);
}

#[test]
fn blend_parity_hard_mix() {
    test_mode(&all_modes()[16]);
}

#[test]
fn blend_parity_subtract() {
    test_mode(&all_modes()[17]);
}

#[test]
fn blend_parity_divide() {
    test_mode(&all_modes()[18]);
}

/// Summary test that runs all modes and prints a consolidated report.
#[test]
fn blend_parity_all_modes_summary() {
    let modes = all_modes();
    let (fg_raw, bg_raw) = make_test_images();
    let fg_ppm = encode_ppm(W, H, &fg_raw);
    let bg_ppm = encode_ppm(W, H, &bg_raw);
    let info = make_image_info(W, H);

    eprintln!("\n=== Blend Mode Parity Summary ===");
    eprintln!(
        "{:15} {:6} {:>8} {:>8}  {}",
        "Mode", "Tool", "MaxDiff", "DiffChs", "Status"
    );
    eprintln!("{}", "-".repeat(55));

    let mut all_pass = true;

    for spec in &modes {
        let ours = blend(&fg_raw, &info, &bg_raw, &info, spec.blend_mode).unwrap();

        let (ref_pixels, ref_tool) = if let Some(vips_num) = spec.vips_num {
            (run_vips(&fg_ppm, &bg_ppm, vips_num), "vips")
        } else {
            (run_magick(&fg_ppm, &bg_ppm, spec.magick_name), "magick")
        };

        let (max_diff, num_diff) = compare_pixels(&ours, &ref_pixels);

        let status = if max_diff == 0 {
            "EXACT"
        } else if max_diff <= TOLERANCE {
            "OK(+/-1)"
        } else {
            all_pass = false;
            "FAIL"
        };

        eprintln!(
            "{:15} {:6} {:>8} {:>8}  {}",
            spec.name, ref_tool, max_diff, num_diff, status
        );
    }

    eprintln!("{}", "=".repeat(55));
    assert!(all_pass, "Some blend modes exceeded tolerance");
}
