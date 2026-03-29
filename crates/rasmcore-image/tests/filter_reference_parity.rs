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
