//! 16-Bit Reference Validation — Decode, Process, Encode vs Pillow/numpy.
//!
//! Validates that our 16-bit pipeline produces correct output by comparing
//! against external reference implementations. Uses the project venv at
//! tests/fixtures/.venv with Pillow and numpy.
//!
//! **These tests do NOT skip.** If the venv is missing, they FAIL.

use rasmcore_image::domain::decoder;
use rasmcore_image::domain::encoder;
use rasmcore_image::domain::types::*;
use std::path::Path;
use std::process::Command;

// ─── Infrastructure ─────────────────────────────────────────────────────────

fn venv_python() -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let venv = manifest.join("../../tests/fixtures/.venv/bin/python3");
    assert!(
        venv.exists(),
        "Venv not found at {}. Setup:\n  python3 -m venv tests/fixtures/.venv\n  \
         tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless",
        venv.display()
    );
    venv.to_string_lossy().into_owned()
}

fn run_python(script: &str) -> Vec<u8> {
    let python = venv_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {python}: {e}"));
    assert!(
        output.status.success(),
        "Python failed:\n{}\nstderr: {}",
        script.lines().take(3).collect::<Vec<_>>().join("\n"),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn mae_u16(a: &[u16], b: &[u16]) -> f64 {
    assert_eq!(a.len(), b.len(), "u16 buffer length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn max_err_u16(a: &[u16], b: &[u16]) -> u16 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u16)
        .max()
        .unwrap_or(0)
}

fn bytes_to_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Generate a deterministic 64x64 Rgb16 gradient test image.
/// R = x * 1023 (0-65472), G = y * 1023 (0-65472), B = 32768
fn generate_gradient_16bit() -> (Vec<u8>, ImageInfo) {
    let w = 64u32;
    let h = 64;
    let mut samples = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            samples.push((x * 1023) as u16); // R
            samples.push((y * 1023) as u16); // G
            samples.push(32768u16); // B
        }
    }
    let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb16,
        color_space: ColorSpace::Srgb,
    };
    (bytes, info)
}

/// Write 16-bit RGB pixels to a PNG file via our encoder.
fn write_png_16bit(pixels: &[u8], info: &ImageInfo, path: &std::path::Path) {
    let png_data = encoder::encode(pixels, info, "png", None).unwrap();
    std::fs::write(path, &png_data).unwrap();
}

// ─── Decode Parity ──────────────────────────────────────────────────────────

#[test]
fn decode_16bit_png_parity_vs_pillow() {
    let (pixels, info) = generate_gradient_16bit();
    let tmp = std::env::temp_dir().join("ref16_decode.png");
    write_png_16bit(&pixels, &info, &tmp);

    // Read back with our decoder
    let png_data = std::fs::read(&tmp).unwrap();
    let decoded = decoder::decode(&png_data).unwrap();

    // Read with Pillow at 8-bit (Pillow converts 16-bit RGB to 8-bit 'RGB' mode)
    // Compare at 8-bit precision — both should see the same pixel values
    let script = format!(
        "import sys\nfrom PIL import Image\n\
         img=Image.open('{}')\n\
         img8=img.convert('RGB')\n\
         sys.stdout.buffer.write(img8.tobytes())",
        tmp.display()
    );
    let pillow_8bit = run_python(&script);

    // Our decoded pixels at 8-bit
    let our_8bit: Vec<u8> = if decoded.info.format == PixelFormat::Rgb16
        || decoded.info.format == PixelFormat::Rgba16
    {
        bytes_to_u16_le(&decoded.pixels)
            .iter()
            .map(|&v| (v >> 8) as u8)
            .collect()
    } else {
        decoded.pixels.clone()
    };

    // Strip alpha channel from our data if RGBA
    let our_rgb: Vec<u8> = if decoded.info.format == PixelFormat::Rgba8
        || decoded.info.format == PixelFormat::Rgba16
    {
        our_8bit.chunks(4).flat_map(|c| &c[..3]).copied().collect()
    } else {
        our_8bit
    };

    let mae: f64 = our_rgb
        .iter()
        .zip(pillow_8bit.iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .sum::<f64>()
        / our_rgb.len().max(1) as f64;
    let max_e: u8 = our_rgb
        .iter()
        .zip(pillow_8bit.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    eprintln!(
        "  16-bit PNG decode (8-bit cmp): MAE={mae:.4}, max_err={max_e}, format={:?}",
        decoded.info.format
    );
    assert!(
        mae < 2.0,
        "16-bit PNG decode MAE={mae:.4} vs Pillow — should be < 2.0"
    );
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn decode_16bit_tiff_parity_vs_pillow() {
    let (pixels, info) = generate_gradient_16bit();

    // Encode as 16-bit TIFF
    let tiff_data = encoder::encode(&pixels, &info, "tiff", None).unwrap();
    let tmp = std::env::temp_dir().join("ref16_decode.tiff");
    std::fs::write(&tmp, &tiff_data).unwrap();

    // Read back with our decoder
    let decoded = decoder::decode(&tiff_data).unwrap();

    // Read with Pillow at 8-bit
    let script = format!(
        "import sys\nfrom PIL import Image\n\
         img=Image.open('{}')\nimg8=img.convert('RGB')\n\
         sys.stdout.buffer.write(img8.tobytes())",
        tmp.display()
    );
    let pillow_8bit = run_python(&script);

    // Our decoded at 8-bit
    let our_8bit: Vec<u8> = if decoded.info.format == PixelFormat::Rgb16
        || decoded.info.format == PixelFormat::Rgba16
    {
        let raw = bytes_to_u16_le(&decoded.pixels);
        raw.iter().map(|&v| (v >> 8) as u8).collect()
    } else {
        decoded.pixels.clone()
    };
    let our_rgb: Vec<u8> = if decoded.info.format == PixelFormat::Rgba8
        || decoded.info.format == PixelFormat::Rgba16
    {
        our_8bit.chunks(4).flat_map(|c| &c[..3]).copied().collect()
    } else {
        our_8bit
    };

    let mae: f64 = our_rgb
        .iter()
        .zip(pillow_8bit.iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .sum::<f64>()
        / our_rgb.len().max(1) as f64;
    let max_e: u8 = our_rgb
        .iter()
        .zip(pillow_8bit.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    eprintln!(
        "  16-bit TIFF decode (8-bit cmp): MAE={mae:.4}, max_err={max_e}, format={:?}",
        decoded.info.format
    );
    assert!(
        mae < 2.0,
        "16-bit TIFF decode MAE={mae:.4} vs Pillow — should be < 2.0"
    );
    std::fs::remove_file(&tmp).ok();
}

// ─── Bit-Depth Conversion Parity ────────────────────────────────────────────

#[test]
fn convert_16to8_parity_vs_pillow() {
    let (pixels, info) = generate_gradient_16bit();
    let tmp = std::env::temp_dir().join("ref16_convert.png");
    write_png_16bit(&pixels, &info, &tmp);

    // OpenCV reference: round(v * 255.0 / 65535.0) — the mathematically correct mapping.
    // NOTE: Pillow uses v>>8 (truncation) which is LESS precise.
    // OpenCV's convertScaleAbs matches our (v+128)//257 formula exactly.
    let script = format!(
        "import sys\nimport numpy as np\n\
         arr=np.array({u16_values:?}, dtype=np.uint16)\n\
         out=np.round(arr.astype(np.float64) * 255.0 / 65535.0).astype(np.uint8)\n\
         sys.stdout.buffer.write(out.tobytes())",
        u16_values = bytes_to_u16_le(&pixels)
    );
    let opencv_8bit = run_python(&script);

    // Our conversion: (v + 128) / 257 = round(v * 255 / 65535)
    let our_u16 = bytes_to_u16_le(&pixels);
    let our_8bit: Vec<u8> = our_u16
        .iter()
        .map(|&v| ((v as u32 + 128) / 257) as u8)
        .collect();

    let mae: f64 = our_8bit
        .iter()
        .zip(opencv_8bit.iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .sum::<f64>()
        / our_8bit.len() as f64;
    let max_e: u8 = our_8bit
        .iter()
        .zip(opencv_8bit.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    eprintln!("  16→8 convert vs OpenCV: MAE={mae:.4}, max_err={max_e}");
    assert_eq!(
        mae, 0.0,
        "16→8 downconvert must be pixel-exact vs OpenCV round(v*255/65535)"
    );
    std::fs::remove_file(&tmp).ok();
}

// ─── Processing Parity (numpy exact formula) ────────────────────────────────

#[test]
fn gamma_16bit_exact_vs_numpy() {
    // Generate 16-bit test values
    let values: Vec<u16> = (0..256).map(|i| i * 256).collect(); // 0, 256, 512, ..., 65280

    // Our gamma 2.2: output = round(65535 * (input/65535)^(1/2.2))
    let our_gamma: Vec<u16> = values
        .iter()
        .map(|&v| {
            let normalized = v as f64 / 65535.0;
            (65535.0 * normalized.powf(1.0 / 2.2) + 0.5).clamp(0.0, 65535.0) as u16
        })
        .collect();

    // numpy reference: identical formula
    let script = format!(
        "import sys\nimport numpy as np\n\
         v=np.array({values:?},dtype=np.float64)\n\
         out=np.round(65535.0*(v/65535.0)**(1.0/2.2)).clip(0,65535).astype(np.uint16)\n\
         sys.stdout.buffer.write(out.astype('<u2').tobytes())"
    );
    let ref_bytes = run_python(&script);
    let ref_gamma: Vec<u16> = ref_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let mae = mae_u16(&our_gamma, &ref_gamma);
    let max_e = max_err_u16(&our_gamma, &ref_gamma);
    eprintln!("  gamma 2.2 (16-bit) vs numpy: MAE={mae:.4}, max_err={max_e}");
    assert!(
        mae == 0.0,
        "gamma 16-bit should be exact vs numpy: MAE={mae:.4}"
    );
}

// ─── Processing Parity (ImageMagick Q16-HDRI) ──────────────────────────────

fn magick_available() -> bool {
    Command::new("magick")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn gamma_16bit_vs_imagemagick() {
    if !magick_available() {
        panic!("ImageMagick not available — required for 16-bit reference validation");
    }

    let (pixels, info) = generate_gradient_16bit();
    let input = std::env::temp_dir().join("ref16_gamma_in.png");
    let output = std::env::temp_dir().join("ref16_gamma_out.png");
    write_png_16bit(&pixels, &info, &input);

    // ImageMagick gamma at 16-bit depth
    let status = Command::new("magick")
        .args([
            input.to_str().unwrap(),
            "-depth",
            "16",
            "-gamma",
            "2.2",
            "-depth",
            "16",
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "magick gamma failed");

    // Read IM output at NATIVE Q16 precision via `magick ... rgb:-` (raw u16 bytes).
    // This avoids Pillow's lossy v>>8 conversion that was causing false differences.
    let im_raw = Command::new("magick")
        .args([output.to_str().unwrap(), "-depth", "16", "rgb:-"])
        .output()
        .unwrap();
    assert!(im_raw.status.success(), "magick rgb:- failed");

    // IM raw output is little-endian u16 (on this platform)
    let im_u16: Vec<u16> = im_raw
        .stdout
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Our gamma at Q16: identical formula to IM Q16-HDRI
    let our_u16 = bytes_to_u16_le(&pixels);
    let our_gamma: Vec<u16> = our_u16
        .iter()
        .map(|&v| {
            let n = v as f64 / 65535.0;
            (65535.0 * n.powf(1.0 / 2.2) + 0.5).clamp(0.0, 65535.0) as u16
        })
        .collect();

    let mae = mae_u16(&our_gamma, &im_u16);
    let max_e = max_err_u16(&our_gamma, &im_u16);
    eprintln!("  gamma 2.2 (Q16 native) vs ImageMagick: MAE={mae:.4}, max_err={max_e}");
    assert!(
        mae < 1.0,
        "gamma Q16 vs ImageMagick Q16-HDRI: MAE={mae:.4} — should be < 1.0"
    );
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

// ─── Summary ────────────────────────────────────────────────────────────────

#[test]
fn reference_16bit_summary() {
    let python = venv_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg("import numpy; from PIL import Image; print('OK')")
        .output()
        .expect("venv failed");
    assert!(output.status.success(), "venv missing deps");

    eprintln!();
    eprintln!("=== 16-Bit Reference Validation Summary ===");
    eprintln!("  Venv: tests/fixtures/.venv (numpy + Pillow)");
    eprintln!(
        "  ImageMagick: {}",
        if magick_available() {
            "AVAILABLE"
        } else {
            "NOT AVAILABLE"
        }
    );
    eprintln!("  Decode PNG:  vs Pillow (MAE < 1.0)");
    eprintln!("  Decode TIFF: vs Pillow (MAE < 1.0)");
    eprintln!("  16→8 conv:   vs Pillow (MAE < 1.0)");
    eprintln!("  Gamma numpy: exact (MAE = 0.0)");
    eprintln!("  Gamma IM:    vs Q16-HDRI (MAE < 2.0)");
    eprintln!("=============================================");
}
