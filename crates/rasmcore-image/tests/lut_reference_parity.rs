//! 3D LUT Reference Parity Tests — .cube vs ffmpeg, HALD vs ImageMagick.
//!
//! Validates that our .cube parser + ColorLut3D::apply produces the same
//! pixel output as ffmpeg's lut3d filter, and our HALD parser matches
//! ImageMagick's -hald-clut operation.

use rasmcore_image::domain::types::*;
use std::process::Command;

fn has_tool(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn mean_absolute_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "buffer length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

/// Make a 32x32 RGB gradient test image (deterministic).
fn make_gradient_image() -> (Vec<u8>, ImageInfo) {
    let w = 32u32;
    let h = 32u32;
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 8) as u8); // R: 0-248
            pixels.push((y * 8) as u8); // G: 0-248
            pixels.push(((x + y) * 4) as u8); // B: 0-248
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };
    (pixels, info)
}

/// Generate a non-trivial .cube LUT (warm color grade: boost R, reduce B).
fn make_warm_cube(grid_size: usize) -> String {
    let mut text = format!("TITLE \"warm test grade\"\nLUT_3D_SIZE {grid_size}\n");
    let scale = 1.0 / (grid_size - 1) as f64;
    for b in 0..grid_size {
        for g in 0..grid_size {
            for r in 0..grid_size {
                let rf = r as f64 * scale;
                let gf = g as f64 * scale;
                let bf = b as f64 * scale;
                // Warm grade: boost red, slight green shift, reduce blue
                let ro = (rf * 1.1 + 0.02).min(1.0);
                let go = gf * 1.0;
                let bo = (bf * 0.85).max(0.0);
                text.push_str(&format!("{ro:.6} {go:.6} {bo:.6}\n"));
            }
        }
    }
    text
}

/// .cube parity vs ffmpeg lut3d filter.
///
/// Apply the same .cube LUT to the same image via both our code and ffmpeg,
/// compare pixel output.
#[test]
fn cube_lut_parity_vs_ffmpeg() {
    if !has_tool("ffmpeg") {
        eprintln!("  cube_lut_parity_vs_ffmpeg: SKIP (ffmpeg not found)");
        return;
    }

    let (pixels, info) = make_gradient_image();
    let cube_text = make_warm_cube(17);

    // Our application
    let lut = rasmcore_image::domain::color_lut::parse_cube_lut(&cube_text).unwrap();
    let our_output = lut.apply(&pixels, &info).unwrap();

    // Write inputs to temp files
    let tmp = std::env::temp_dir().join("rasmcore_cube_parity");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let input_png = tmp.join("input.png");
    let cube_file = tmp.join("warm.cube");
    let ffmpeg_out = tmp.join("ffmpeg_out.png");

    let encoded = rasmcore_image::domain::encoder::encode(&pixels, &info, "png", None).unwrap();
    std::fs::write(&input_png, &encoded).unwrap();
    std::fs::write(&cube_file, &cube_text).unwrap();

    // Apply via ffmpeg
    let output = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input_png.to_str().unwrap(),
            "-vf",
            &format!("lut3d={}", cube_file.to_str().unwrap()),
            ffmpeg_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    if !output.status.success() {
        eprintln!(
            "  ffmpeg lut3d failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let _ = std::fs::remove_dir_all(&tmp);
        return;
    }

    // Decode ffmpeg output
    let ffmpeg_data = std::fs::read(&ffmpeg_out).unwrap();
    let ffmpeg_decoded = rasmcore_image::domain::decoder::decode(&ffmpeg_data).unwrap();

    // Compare — both should be RGB8 32x32
    assert_eq!(ffmpeg_decoded.info.width, info.width);
    assert_eq!(ffmpeg_decoded.info.height, info.height);

    // ffmpeg might output RGBA or RGB — normalize to RGB for comparison
    let ffmpeg_rgb = if ffmpeg_decoded.info.format == PixelFormat::Rgba8 {
        ffmpeg_decoded
            .pixels
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<u8>>()
    } else {
        ffmpeg_decoded.pixels.clone()
    };

    assert_eq!(
        our_output.len(),
        ffmpeg_rgb.len(),
        "output size mismatch: ours={} ffmpeg={}",
        our_output.len(),
        ffmpeg_rgb.len()
    );

    let mae = mean_absolute_error(&our_output, &ffmpeg_rgb);
    eprintln!("  .cube LUT parity: MAE={mae:.4} (ours vs ffmpeg lut3d)");
    assert!(
        mae < 1.5,
        ".cube LUT parity too high: MAE={mae:.4} (expected < 1.5 for tetrahedral interp differences)"
    );

    let _ = std::fs::remove_dir_all(&tmp);
    eprintln!("  cube_lut_parity_vs_ffmpeg: PASS");
}

/// HALD CLUT parity vs ImageMagick -hald-clut.
///
/// Generate a graded HALD via IM, apply to a test image via both IM and our code,
/// compare pixel output.
#[test]
fn hald_lut_parity_vs_imagemagick() {
    if !has_tool("magick") {
        eprintln!("  hald_lut_parity_vs_imagemagick: SKIP (magick not found)");
        return;
    }

    let tmp = std::env::temp_dir().join("rasmcore_hald_parity");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let (pixels, info) = make_gradient_image();
    let input_png = tmp.join("input.png");
    let encoded = rasmcore_image::domain::encoder::encode(&pixels, &info, "png", None).unwrap();
    std::fs::write(&input_png, &encoded).unwrap();

    // Generate identity HALD level 2 (8x8 image, 4^3=64 LUT entries) — small for test speed
    // Use level 4 (64x64 image, 16^3=4096 LUT entries) for better accuracy
    let hald_identity = tmp.join("hald_identity.png");
    let hald_graded = tmp.join("hald_graded.png");
    let im_output = tmp.join("im_output.png");

    // Generate identity HALD
    let out = Command::new("magick")
        .args(["-size", "4", "hald:", hald_identity.to_str().unwrap()])
        .output()
        .unwrap();
    if !out.status.success() {
        eprintln!(
            "  magick hald: failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_dir_all(&tmp);
        return;
    }

    // Apply a color modification to create graded HALD (warm: boost saturation, shift hue)
    let out = Command::new("magick")
        .args([
            hald_identity.to_str().unwrap(),
            "-modulate",
            "100,130,95",
            "-depth",
            "8",
            hald_graded.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    if !out.status.success() {
        eprintln!(
            "  magick modulate failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_dir_all(&tmp);
        return;
    }

    // Apply graded HALD to input via ImageMagick
    let out = Command::new("magick")
        .args([
            input_png.to_str().unwrap(),
            hald_graded.to_str().unwrap(),
            "-hald-clut",
            "-depth",
            "8",
            im_output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    if !out.status.success() {
        eprintln!(
            "  magick -hald-clut failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_dir_all(&tmp);
        return;
    }

    // Apply graded HALD via our code
    let hald_data = std::fs::read(&hald_graded).unwrap();
    let hald_decoded = rasmcore_image::domain::decoder::decode(&hald_data).unwrap();

    // Convert HALD to RGB8 if needed
    let hald_rgb = if hald_decoded.info.format == PixelFormat::Rgba8 {
        let rgb: Vec<u8> = hald_decoded
            .pixels
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        let rgb_info = ImageInfo {
            width: hald_decoded.info.width,
            height: hald_decoded.info.height,
            format: PixelFormat::Rgb8,
            color_space: hald_decoded.info.color_space,
        };
        (rgb, rgb_info)
    } else {
        (hald_decoded.pixels.clone(), hald_decoded.info.clone())
    };

    let our_lut =
        rasmcore_image::domain::color_lut::parse_hald_lut(&hald_rgb.0, &hald_rgb.1).unwrap();
    let our_output = our_lut.apply(&pixels, &info).unwrap();

    // Decode IM output
    let im_data = std::fs::read(&im_output).unwrap();
    let im_decoded = rasmcore_image::domain::decoder::decode(&im_data).unwrap();
    let im_rgb = if im_decoded.info.format == PixelFormat::Rgba8 {
        im_decoded
            .pixels
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<u8>>()
    } else {
        im_decoded.pixels.clone()
    };

    assert_eq!(
        our_output.len(),
        im_rgb.len(),
        "output size mismatch: ours={} IM={}",
        our_output.len(),
        im_rgb.len()
    );

    let mae = mean_absolute_error(&our_output, &im_rgb);
    eprintln!("  HALD CLUT parity: MAE={mae:.4} (ours vs IM -hald-clut)");
    assert!(
        mae < 2.0,
        "HALD CLUT parity too high: MAE={mae:.4} (expected < 2.0 — interpolation method differences)"
    );

    let _ = std::fs::remove_dir_all(&tmp);
    eprintln!("  hald_lut_parity_vs_imagemagick: PASS");
}

/// .cube identity LUT via ffmpeg: verify ffmpeg also treats it as identity.
#[test]
fn cube_identity_parity_vs_ffmpeg() {
    if !has_tool("ffmpeg") {
        eprintln!("  cube_identity_parity_vs_ffmpeg: SKIP (ffmpeg not found)");
        return;
    }

    let (pixels, info) = make_gradient_image();
    let n = 17;
    let mut cube_text = format!("LUT_3D_SIZE {n}\n");
    let scale = 1.0 / (n - 1) as f64;
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                cube_text.push_str(&format!(
                    "{:.6} {:.6} {:.6}\n",
                    r as f64 * scale,
                    g as f64 * scale,
                    b as f64 * scale
                ));
            }
        }
    }

    // Our application — should be near-identity
    let lut = rasmcore_image::domain::color_lut::parse_cube_lut(&cube_text).unwrap();
    let our_output = lut.apply(&pixels, &info).unwrap();

    let tmp = std::env::temp_dir().join("rasmcore_cube_identity");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let input_png = tmp.join("input.png");
    let cube_file = tmp.join("identity.cube");
    let ffmpeg_out = tmp.join("ffmpeg_out.png");

    let encoded = rasmcore_image::domain::encoder::encode(&pixels, &info, "png", None).unwrap();
    std::fs::write(&input_png, &encoded).unwrap();
    std::fs::write(&cube_file, &cube_text).unwrap();

    let output = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input_png.to_str().unwrap(),
            "-vf",
            &format!("lut3d={}", cube_file.to_str().unwrap()),
            ffmpeg_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    if !output.status.success() {
        let _ = std::fs::remove_dir_all(&tmp);
        return;
    }

    let ffmpeg_data = std::fs::read(&ffmpeg_out).unwrap();
    let ffmpeg_decoded = rasmcore_image::domain::decoder::decode(&ffmpeg_data).unwrap();
    let ffmpeg_rgb = if ffmpeg_decoded.info.format == PixelFormat::Rgba8 {
        ffmpeg_decoded
            .pixels
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<u8>>()
    } else {
        ffmpeg_decoded.pixels.clone()
    };

    // Both should be near-identity — compare them to each other
    let mae_us_vs_ff = mean_absolute_error(&our_output, &ffmpeg_rgb);
    // And both should be near the original
    let mae_us_vs_orig = mean_absolute_error(&our_output, &pixels);
    let mae_ff_vs_orig = mean_absolute_error(&ffmpeg_rgb, &pixels);

    eprintln!(
        "  Identity .cube: us_vs_ffmpeg={mae_us_vs_ff:.4}, us_vs_orig={mae_us_vs_orig:.4}, ff_vs_orig={mae_ff_vs_orig:.4}"
    );
    assert!(
        mae_us_vs_ff < 1.0,
        "identity LUT: our vs ffmpeg should be near-zero, got {mae_us_vs_ff}"
    );
    assert!(
        mae_us_vs_orig < 1.0,
        "identity LUT: our output should match input, got {mae_us_vs_orig}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
    eprintln!("  cube_identity_parity_vs_ffmpeg: PASS");
}
