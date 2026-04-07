//! ML Upscale PoC — Real-ESRGAN x4plus via ONNX Runtime
//!
//! Validates the ML inference execution path:
//! 1. Downloads Real-ESRGAN ONNX model (~64MB) from HuggingFace
//! 2. Loads input image (PNG/JPEG)
//! 3. Converts RGBA HWC → RGB NCHW float32 [0,1]
//! 4. Runs inference (auto-detects CoreML/CUDA/CPU)
//! 5. Converts output RGB NCHW → RGBA HWC u8
//! 6. Saves 4x upscaled output
//!
//! Usage:
//!   # First, install ONNX Runtime shared library:
//!   #   macOS: brew install onnxruntime
//!   #
//!   # Set the dynamic library path:
//!   #   export ORT_DYLIB_PATH=$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib
//!   #
//!   cargo run -- input.png output.png

use std::path::{Path, PathBuf};
use std::time::Instant;

fn model_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("rasmcore").join("models")
}

const MODEL_URL: &str = "https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx";
const MODEL_FILENAME: &str = "Real-ESRGAN-x4plus.onnx";

fn download_model(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if dest.exists() {
        let size = std::fs::metadata(dest)?.len();
        if size > 1_000_000 {
            println!("  Model cached at {} ({:.1} MB)", dest.display(), size as f64 / 1_048_576.0);
            return Ok(());
        }
    }

    println!("  Downloading model from {url}");
    std::fs::create_dir_all(dest.parent().unwrap())?;

    let status = std::process::Command::new("curl")
        .args(["-L", "-o", &dest.to_string_lossy(), "--progress-bar", url])
        .status()?;

    if !status.success() {
        return Err(format!("curl failed with status {status}").into());
    }

    let size = std::fs::metadata(dest)?.len();
    println!("  Downloaded {:.1} MB", size as f64 / 1_048_576.0);
    Ok(())
}

/// Convert RGBA u8 image to RGB f32 NCHW flat vec [1, 3, H, W] in [0, 1].
fn image_to_nchw(img: &image::RgbaImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let plane = (h * w) as usize;
    let mut tensor = vec![0.0f32; 3 * plane];

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y);
            let idx = y as usize * w as usize + x as usize;
            tensor[idx] = px[0] as f32 / 255.0;
            tensor[plane + idx] = px[1] as f32 / 255.0;
            tensor[2 * plane + idx] = px[2] as f32 / 255.0;
        }
    }

    tensor
}

/// Convert RGB f32 NCHW flat slice [1, 3, H, W] to RGBA u8 image.
fn nchw_to_image(data: &[f32], out_h: u32, out_w: u32) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(out_w, out_h);
    let plane = (out_h * out_w) as usize;

    for y in 0..out_h {
        for x in 0..out_w {
            let idx = y as usize * out_w as usize + x as usize;
            let r = (data[idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let g = (data[plane + idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let b = (data[2 * plane + idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }

    img
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.png> <output.png>", args[0]);
        eprintln!();
        eprintln!("Environment:");
        eprintln!("  ORT_DYLIB_PATH  Path to libonnxruntime.dylib/.so");
        eprintln!("                  macOS: $(brew --prefix onnxruntime)/lib/libonnxruntime.dylib");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let total_start = Instant::now();

    // ── Step 1: Download model ─────────────────────────────────────────
    println!("Step 1: Model");
    let t = Instant::now();
    let model_path = model_cache_dir().join(MODEL_FILENAME);
    download_model(MODEL_URL, &model_path)?;
    println!("  Time: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    // ── Step 2: Load image ─────────────────────────────────────────────
    println!("Step 2: Load image");
    let t = Instant::now();
    let img = image::open(input_path)?.to_rgba8();
    let (w, h) = img.dimensions();
    println!("  Size: {w}x{h}");
    println!("  Time: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    // ── Step 3: Create ORT session ─────────────────────────────────────
    println!("Step 3: Create ONNX Runtime session");
    let t = Instant::now();

    ort::init().commit();

    let mut session = ort::session::Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .commit_from_file(&model_path)?;

    println!("  Input count: {}", session.inputs().len());
    println!("  Output count: {}", session.outputs().len());
    println!("  Session load time: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    // ── Step 4: Preprocess ─────────────────────────────────────────────
    println!("Step 4: Preprocess (RGBA HWC u8 -> RGB NCHW f32)");
    let t = Instant::now();
    let tensor_data = image_to_nchw(&img);
    let input_tensor = ort::value::Tensor::from_array(([1usize, 3, h as usize, w as usize], tensor_data))?;
    println!("  Tensor shape: [1, 3, {h}, {w}]");
    println!("  Preprocess time: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    // ── Step 5: Run inference ──────────────────────────────────────────
    println!("Step 5: Run inference");
    let t = Instant::now();
    let outputs = session.run(ort::inputs![input_tensor])?;
    let inference_time = t.elapsed().as_secs_f64() * 1000.0;

    let output_value = &outputs[0];
    let (shape, output_data) = output_value.try_extract_tensor::<f32>()?;
    println!("  Output shape: {:?}", &shape[..]);
    println!("  Inference time: {:.1}ms", inference_time);

    let out_h = shape[2] as u32;
    let out_w = shape[3] as u32;

    // ── Step 6: Postprocess and save ───────────────────────────────────
    println!("Step 6: Postprocess (RGB NCHW f32 -> RGBA HWC u8) and save");
    let t = Instant::now();
    let output_img = nchw_to_image(output_data, out_h, out_w);
    output_img.save(output_path)?;
    println!("  Output: {out_w}x{out_h} (4x upscale)");
    println!("  Postprocess + save time: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    println!("\nTotal time: {:.1}ms", total_start.elapsed().as_secs_f64() * 1000.0);
    println!("Output saved to: {output_path}");

    Ok(())
}
