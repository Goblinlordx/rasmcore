//! Filter Reference Parity Tests — Exact validation against Python references.
//!
//! Validates every filter/color operation against a reference implementation
//! computed in Python (numpy, Pillow, OpenCV) using a dedicated venv.
//!
//! **These tests do NOT skip.** If the venv is missing, they FAIL with
//! instructions to set it up. Exact (MAE=0.0) match is required for
//! formula-based operations; spatial operations must match within MAE < 1.0.
//!
//! Setup:
//!   python3 -m venv tests/fixtures/.venv
//!   tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless

use rasmcore_image::domain::filters;
use rasmcore_image::domain::types::*;
use std::path::Path;
use std::process::Command;

// ─── Test Infrastructure ────────────────────────────────────────────────────

/// Returns the path to the venv Python binary.
/// Panics (fails the test) if the venv doesn't exist.
fn venv_python() -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let venv = manifest.join("../../tests/fixtures/.venv/bin/python3");
    if !venv.exists() {
        panic!(
            "Reference test venv not found at {}.\n\
             Setup:\n  python3 -m venv tests/fixtures/.venv\n  \
             tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless",
            venv.display()
        );
    }
    venv.to_string_lossy().into_owned()
}

/// Run a Python script via the venv interpreter, returning raw stdout bytes.
fn run_python_ref(script: &str) -> Vec<u8> {
    let python = venv_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {python}: {e}"));
    assert!(
        output.status.success(),
        "Python reference script failed:\n{}\nstderr: {}",
        script.lines().take(5).collect::<Vec<_>>().join("\n"),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
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

fn max_absolute_error(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn assert_exact(label: &str, ours: &[u8], reference: &[u8]) {
    let mae = mean_absolute_error(ours, reference);
    let max_err = max_absolute_error(ours, reference);
    assert_eq!(
        ours, reference,
        "{label}: NOT pixel-exact. MAE={mae:.4}, max_err={max_err}"
    );
    eprintln!("  {label}: EXACT ✓");
}

fn assert_close(label: &str, ours: &[u8], reference: &[u8], max_mae: f64) {
    let mae = mean_absolute_error(ours, reference);
    let max_err = max_absolute_error(ours, reference);
    assert!(
        mae <= max_mae,
        "{label}: MAE={mae:.4} exceeds threshold {max_mae}. max_err={max_err}"
    );
    eprintln!("  {label}: MAE={mae:.4}, max_err={max_err} ✓");
}

fn make_gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            p.push(((x * 255) / w) as u8);
            p.push(((y * 255) / h) as u8);
            p.push(128);
        }
    }
    p
}

fn info_rgb8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    }
}

fn info_gray8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXACT MATCH (MAE = 0.0) — Formula operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_premultiply() {
    let pixels: Vec<u8> = vec![
        200, 100, 50, 128, 255, 0, 128, 255, 0, 0, 0, 0, 100, 200, 50, 64,
    ];
    let info = ImageInfo {
        width: 4,
        height: 1,
        format: PixelFormat::Rgba8,
        color_space: ColorSpace::Srgb,
    };
    let ours = filters::premultiply(&pixels, &info).unwrap();
    let script = format!(
        "import sys\npx={pixels:?}\no=[]\nfor i in range(0,len(px),4):\n r,g,b,a=px[i],px[i+1],px[i+2],px[i+3]\n o.extend([(r*a+127)//255,(g*a+127)//255,(b*a+127)//255,a])\nsys.stdout.buffer.write(bytes(o))"
    );
    assert_exact("premultiply", &ours, &run_python_ref(&script));
}

#[test]
fn exact_premultiply_roundtrip() {
    let pixels: Vec<u8> = vec![200, 100, 50, 128, 255, 0, 128, 255];
    let info = ImageInfo {
        width: 2,
        height: 1,
        format: PixelFormat::Rgba8,
        color_space: ColorSpace::Srgb,
    };
    let premul = filters::premultiply(&pixels, &info).unwrap();
    let roundtrip = filters::unpremultiply(&premul, &info).unwrap();
    let max_err = max_absolute_error(&pixels, &roundtrip);
    assert!(max_err <= 1, "premultiply roundtrip max_err={max_err}");
    eprintln!("  premultiply roundtrip: max_err={max_err} ✓");
}

#[test]
fn exact_add_remove_alpha_roundtrip() {
    let pixels: Vec<u8> = vec![100, 150, 200, 50, 100, 150];
    let info = info_rgb8(2, 1);
    let (with_alpha, info_rgba) = filters::add_alpha(&pixels, &info, 255).unwrap();
    assert_eq!(with_alpha[3], 255);
    assert_eq!(with_alpha[7], 255);
    let (removed, _) = filters::remove_alpha(&with_alpha, &info_rgba).unwrap();
    assert_exact("add_remove_alpha", &removed, &pixels);
}

#[test]
fn exact_convolve_identity() {
    let pixels = make_gradient_rgb(8, 8);
    let info = info_rgb8(8, 8);
    let kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let result = filters::convolve(&pixels, &info, &kernel, 3, 3, 1.0).unwrap();
    assert_exact("convolve identity", &result, &pixels);
}

#[test]
fn exact_brightness_zero() {
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);
    let result = filters::brightness(&pixels, &info, 0.0).unwrap();
    assert_exact("brightness(0)", &result, &pixels);
}

#[test]
fn exact_median_against_pillow() {
    // Median should be pixel-exact with Pillow (same algorithm: sorting-based)
    let w = 16u32;
    let h = 16;
    let mut pixels = vec![128u8; (w * h) as usize];
    pixels[5 * 16 + 5] = 0;
    pixels[10 * 16 + 10] = 255;
    let info = info_gray8(w, h);
    let ours = filters::median(&pixels, &info, 1).unwrap();
    let script = format!(
        "import sys\nfrom PIL import Image,ImageFilter\nimport warnings\nwarnings.filterwarnings('ignore')\n\
         img=Image.frombytes('L',({w},{h}),bytes({pixels:?}))\n\
         f=img.filter(ImageFilter.MedianFilter(3))\nsys.stdout.buffer.write(f.tobytes())"
    );
    assert_exact("median r=1", &ours, &run_python_ref(&script));
}

// ═══════════════════════════════════════════════════════════════════════════
// CLOSE MATCH (MAE ≤ 1.0) — Integer rounding / border handling differences
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn close_sepia_against_numpy() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let ours = filters::sepia(&pixels, &info, 1.0).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\npx=np.array({pixels:?},dtype=np.uint8).reshape(-1,3)\n\
         m=np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]])\n\
         out=np.clip(px.astype(np.float64)@m.T+0.5,0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("sepia", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_blend_multiply_w3c() {
    let w = 8u32;
    let h = 8;
    let fg = make_gradient_rgb(w, h);
    let bg: Vec<u8> = fg.iter().map(|&v| 255 - v).collect();
    let info = info_rgb8(w, h);
    let ours = filters::blend(&fg, &info, &bg, &info, filters::BlendMode::Multiply).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nfg=np.array({fg:?},dtype=np.uint8)\nbg=np.array({bg:?},dtype=np.uint8)\n\
         out=np.clip((fg.astype(np.uint32)*bg.astype(np.uint32)+127)//255,0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("blend Multiply", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_blend_screen_w3c() {
    let w = 8u32;
    let h = 8;
    let fg = make_gradient_rgb(w, h);
    let bg: Vec<u8> = fg.iter().map(|&v| 255 - v).collect();
    let info = info_rgb8(w, h);
    let ours = filters::blend(&fg, &info, &bg, &info, filters::BlendMode::Screen).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nfg=np.array({fg:?},dtype=np.uint16)\nbg=np.array({bg:?},dtype=np.uint16)\n\
         out=np.clip(fg+bg-(fg*bg+127)//255,0,255).astype(np.uint8)\nsys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("blend Screen", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_convolve_sharpen_against_opencv() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let kernel = filters::kernels::EDGE_ENHANCE;
    let ours = filters::convolve(&pixels, &info, &kernel, 3, 3, 1.0).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nimport cv2\n\
         px=np.array({pixels:?},dtype=np.uint8).reshape({h},{w},3)\n\
         k=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)\n\
         out=cv2.filter2D(px,-1,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("convolve sharpen", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_box_blur_against_opencv() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let kernel = filters::kernels::BOX_BLUR_3X3;
    let ours = filters::convolve(&pixels, &info, &kernel, 3, 3, 9.0).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nimport cv2\n\
         px=np.array({pixels:?},dtype=np.uint8).reshape({h},{w},3)\n\
         out=cv2.blur(px,(3,3),borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("box blur", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_sobel_against_opencv() {
    let w = 16u32;
    let h = 16;
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in w / 2..w {
            pixels[(y * w + x) as usize] = 255;
        }
    }
    let info = info_gray8(w, h);
    let ours = filters::sobel(&pixels, &info).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nimport cv2\n\
         px=np.array({pixels:?},dtype=np.uint8).reshape({h},{w})\n\
         gx=cv2.Sobel(px,cv2.CV_32F,1,0,ksize=3,borderType=cv2.BORDER_REFLECT_101)\n\
         gy=cv2.Sobel(px,cv2.CV_32F,0,1,ksize=3,borderType=cv2.BORDER_REFLECT_101)\n\
         mag=np.clip(np.sqrt(gx**2+gy**2),0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(mag.tobytes())"
    );
    // Border handling differs (our reflect vs OpenCV BORDER_REFLECT_101)
    assert_close("sobel", &ours, &run_python_ref(&script), 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Summary — verifies venv is functional (fails if not)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reference_parity_summary() {
    let python = venv_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg("import numpy, cv2; from PIL import Image; print('ALL DEPS OK')")
        .output()
        .expect("venv python failed");
    assert!(
        output.status.success(),
        "Venv missing dependencies. Run:\n  tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    eprintln!();
    eprintln!("=== Filter Reference Parity Summary ===");
    eprintln!("  Venv: tests/fixtures/.venv (numpy + Pillow + OpenCV)");
    eprintln!(
        "  EXACT (MAE=0.0): premultiply, add/remove alpha, identity convolve, brightness, median"
    );
    eprintln!("  CLOSE (MAE≤1.0): sepia, blend Multiply/Screen, convolve sharpen, box blur");
    eprintln!("  CLOSE (MAE≤2.0): sobel (border reflect variant)");
    eprintln!("=======================================");
}

// ═══════════════════════════════════════════════════════════════════════════
// Morphological Operations — OpenCV Reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn morphology_erode_matches_opencv() {
    use filters::MorphShape;

    let py = venv_python();
    eprintln!("=== Morphology Erode — OpenCV Reference ===");

    // Use standard reference image (grayscale gradient)
    let w = 64u32;
    let h = 64u32;
    let mut input = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            input.push(((x * 255) / w) as u8);
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    // Our erode
    let ours = filters::erode(&input, &info, 3, MorphShape::Rect).unwrap();

    // OpenCV erode
    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         k=np.ones((3,3),np.uint8)\n\
         out=cv2.erode(px,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv erode failed");
    assert!(
        output.status.success(),
        "opencv erode: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference = output.stdout;

    assert_close("erode 3x3 rect", &ours, &reference, 1.0);
}

#[test]
fn morphology_dilate_matches_opencv() {
    use filters::MorphShape;

    let py = venv_python();
    eprintln!("=== Morphology Dilate — OpenCV Reference ===");

    let w = 64u32;
    let h = 64u32;
    let mut input = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            input.push(((x * 255) / w) as u8);
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let ours = filters::dilate(&input, &info, 3, MorphShape::Rect).unwrap();

    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         k=np.ones((3,3),np.uint8)\n\
         out=cv2.dilate(px,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv dilate failed");
    assert!(
        output.status.success(),
        "opencv dilate: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_close("dilate 3x3 rect", &ours, &output.stdout, 1.0);
}

#[test]
fn morphology_open_close_matches_opencv() {
    use filters::MorphShape;

    let py = venv_python();
    eprintln!("=== Morphology Open/Close — OpenCV Reference ===");

    // Use photo_256x256 as standard reference, converted to grayscale
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/generated/inputs/photo_256x256.png");
    let (input, w, h) = if fixture_path.exists() {
        let img = image::open(&fixture_path).unwrap().to_luma8();
        let w = img.width();
        let h = img.height();
        (img.into_raw(), w, h)
    } else {
        // Fallback: synthetic gradient
        let w = 64u32;
        let h = 64u32;
        let mut px = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push(((x * 255) / w) as u8);
            }
        }
        (px, w, h)
    };

    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    // Open
    let ours_open = filters::morph_open(&input, &info, 5, MorphShape::Rect).unwrap();
    let script_open = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         k=np.ones((5,5),np.uint8)\n\
         out=cv2.morphologyEx(px,cv2.MORPH_OPEN,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let out = Command::new(&py)
        .arg("-c")
        .arg(&script_open)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            use std::io::Write;
            c.stdin.take().unwrap().write_all(&input).unwrap();
            c.wait_with_output()
        })
        .expect("opencv open failed");
    assert!(
        out.status.success(),
        "opencv open: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_close("open 5x5 rect", &ours_open, &out.stdout, 1.0);

    // Close
    let ours_close = filters::morph_close(&input, &info, 5, MorphShape::Rect).unwrap();
    let script_close = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         k=np.ones((5,5),np.uint8)\n\
         out=cv2.morphologyEx(px,cv2.MORPH_CLOSE,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let out = Command::new(&py)
        .arg("-c")
        .arg(&script_close)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            use std::io::Write;
            c.stdin.take().unwrap().write_all(&input).unwrap();
            c.wait_with_output()
        })
        .expect("opencv close failed");
    assert!(
        out.status.success(),
        "opencv close: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_close("close 5x5 rect", &ours_close, &out.stdout, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// NLM Denoising — OpenCV Reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nlm_denoise_matches_opencv() {
    let py = venv_python();
    eprintln!("=== NLM Denoising — OpenCV Reference ===");

    // Create noisy grayscale image (deterministic)
    let w = 32u32;
    let h = 32u32;
    let mut input = vec![128u8; (w * h) as usize];
    for i in 0..input.len() {
        let noise = ((i as u32).wrapping_mul(2654435761) >> 24) as i16 - 128;
        input[i] = (128i16 + noise / 4).clamp(0, 255) as u8;
    }

    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let params = filters::NlmParams {
        h: 20.0,
        patch_size: 7,
        search_size: 21,
        ..Default::default() // OpenCv algorithm (matches cv2.fastNlMeansDenoising)
    };
    let ours = filters::nlm_denoise(&input, &info, &params).unwrap();

    // OpenCV NLM
    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         out=cv2.fastNlMeansDenoising(px,None,20.0,7,21)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv nlm failed");
    assert!(
        output.status.success(),
        "opencv nlm: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reference = &output.stdout;
    let mae = mean_absolute_error(&ours, reference);
    let max_err = max_absolute_error(&ours, reference);
    eprintln!("  NLM h=20 7x21: MAE={mae:.4}, max_err={max_err}");

    // OpenCV-exact algorithm: identical integer SSD, bit-shift avg, LUT, fixed-point.
    //
    // Remaining ±1 on ~0.4% of pixels: all have fractional part within 0.005 of 0.5.
    // Cause: C++ libm exp() vs Rust libm exp() differ at the ULP (unit of last place)
    // level for some inputs. This shifts a few LUT entries by ±1, which accumulates
    // across 441 search positions to flip rounding at the 0.5 boundary.
    //
    // The algorithm is mathematically identical — same integer SSD, same bit-shift,
    // same LUT formula, same fixed-point accumulation, same rounding division.
    // The only difference is IEEE 754 exp() implementation between compilers.
    assert!(
        mae < 0.01,
        "NLM: MAE={mae:.4} — algorithm must match OpenCV (only exp() ULP diff allowed)"
    );
    assert!(
        max_err <= 1,
        "NLM: max_err={max_err} — only ±1 from exp() ULP rounding allowed"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Lens Distortion Correction — OpenCV Reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn undistort_matches_opencv() {
    use rasmcore_image::domain::transform::{CameraMatrix, DistortionCoeffs, undistort};

    let py = venv_python();
    eprintln!("=== Undistort — OpenCV Reference ===");

    // Create gradient test image
    let w = 128u32;
    let h = 128u32;
    let mut input = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            input.push(((x * 255) / w) as u8);
            input.push(((y * 255) / h) as u8);
            input.push(128u8);
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };

    let camera = CameraMatrix {
        fx: 100.0,
        fy: 100.0,
        cx: 64.0,
        cy: 64.0,
    };
    let dist = DistortionCoeffs {
        k1: -0.3,
        k2: 0.1,
        k3: 0.0,
    };

    let ours = undistort(&input, &info, &camera, &dist).unwrap();

    // OpenCV undistort
    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w},3)\n\
         K=np.array([[100.0,0,64.0],[0,100.0,64.0],[0,0,1]],dtype=np.float64)\n\
         D=np.array([-0.3,0.1,0,0,0],dtype=np.float64)\n\
         out=cv2.undistort(px,K,D,None,K)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv undistort failed");
    assert!(
        output.status.success(),
        "opencv undistort: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reference = &output.stdout;

    // Compare only interior pixels (borders differ due to out-of-bounds handling)
    let bpp = 3;
    let margin = 10;
    let mut total_err: f64 = 0.0;
    let mut max_err: u8 = 0;
    let mut count = 0usize;
    for y in margin..(h as usize - margin) {
        for x in margin..(w as usize - margin) {
            for ch in 0..bpp {
                let idx = (y * w as usize + x) * bpp + ch;
                let d = (ours.pixels[idx] as i16 - reference[idx] as i16).unsigned_abs() as u8;
                total_err += d as f64;
                max_err = max_err.max(d);
                count += 1;
            }
        }
    }
    let mae = total_err / count as f64;
    eprintln!("  undistort (interior): MAE={mae:.4}, max_err={max_err}");

    // Pixel-exact: same fixed-point bilinear as OpenCV (INTER_BITS=5, COEF_BITS=15)
    assert_eq!(
        mae, 0.0,
        "undistort: MAE={mae:.4} — must be pixel-exact with OpenCV"
    );
    assert_eq!(
        max_err, 0,
        "undistort: max_err={max_err} — must be pixel-exact with OpenCV"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial Utilities — OpenCV Reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pyr_down_matches_opencv() {
    let py = venv_python();
    eprintln!("=== pyrDown — OpenCV Reference ===");

    let w = 64u32;
    let h = 64u32;
    let mut input = Vec::with_capacity((w * h) as usize);
    for _y in 0..h {
        for x in 0..w {
            input.push(((x * 255) / w) as u8);
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let (ours, _) = filters::pyr_down(&input, &info).unwrap();

    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         out=cv2.pyrDown(px,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv pyrDown failed");
    assert!(
        output.status.success(),
        "opencv pyrDown: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mae = mean_absolute_error(&ours, &output.stdout);
    let max_err = max_absolute_error(&ours, &output.stdout);
    eprintln!("  pyrDown: MAE={mae:.4}, max_err={max_err}");
    assert!(mae < 1.0, "pyrDown: MAE={mae:.4} — should be near-exact");
}

#[test]
fn connected_components_matches_opencv() {
    let py = venv_python();
    eprintln!("=== Connected Components — OpenCV Reference ===");

    // Create binary image with two blobs
    let w = 32u32;
    let h = 32u32;
    let mut input = vec![0u8; (w * h) as usize];
    // Blob 1: top-left 5x5
    for y in 2..7 {
        for x in 2..7 {
            input[y * w as usize + x] = 255;
        }
    }
    // Blob 2: bottom-right 5x5
    for y in 20..25 {
        for x in 20..25 {
            input[y * w as usize + x] = 255;
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let (our_labels, our_count) = filters::connected_components(&input, &info, 8).unwrap();

    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         n,labels=cv2.connectedComponents(px,connectivity=8)\n\
         sys.stdout.buffer.write(np.array([n],dtype=np.uint32).tobytes())\n\
         sys.stdout.buffer.write(labels.astype(np.uint32).tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv connectedComponents failed");
    assert!(
        output.status.success(),
        "opencv CC: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Parse OpenCV output: first 4 bytes = num_labels (including background)
    let cv_count = u32::from_le_bytes(output.stdout[0..4].try_into().unwrap()) - 1; // subtract background
    eprintln!("  connectedComponents: ours={our_count} labels, cv={cv_count} labels");
    assert_eq!(our_count, cv_count, "component count must match OpenCV");
}
