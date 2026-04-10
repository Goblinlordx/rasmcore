//! GPU vs CPU Parity Tests — validates GPU shader output matches CPU output.
//!
//! For each GPU-accelerated filter, runs both CPU and GPU paths on the same
//! input and compares outputs with per-channel tolerance. Tests skip gracefully
//! on machines without a GPU.

use rasmcore_image::domain::color_lut::ColorLut3D;
use rasmcore_image::domain::filter_traits::CpuFilter;
use rasmcore_image::domain::filters;
use rasmcore_image::domain::types::*;
use rasmcore_pipeline::Rect;
use rasmcore_pipeline::gpu::{GpuCapable, GpuError, GpuExecutor, GpuOp};
use std::collections::HashMap;

// ─── Inline WgpuExecutor (mirrors rasmcore-cli gpu_executor.rs) ────────────

fn shader_hash(source: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in source.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

struct TestGpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_cache: std::cell::RefCell<HashMap<u64, wgpu::ShaderModule>>,
    adapter_name: String,
    max_buffer: usize,
}

impl TestGpuExecutor {
    fn try_new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| GpuError::NotAvailable("no GPU adapter found".into()))?;

        let adapter_name = adapter.get_info().name.clone();
        let max_buffer = adapter.limits().max_buffer_size as usize;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu-parity-test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .map_err(|e| GpuError::NotAvailable(format!("device creation failed: {e}")))?;

        Ok(Self {
            device,
            queue,
            shader_cache: std::cell::RefCell::new(HashMap::new()),
            adapter_name,
            max_buffer,
        })
    }

    fn get_shader(&self, source: &str) -> wgpu::ShaderModule {
        let hash = shader_hash(source);
        let mut cache = self.shader_cache.borrow_mut();
        if let Some(module) = cache.get(&hash) {
            return module.clone();
        }
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu-test-compute"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        cache.insert(hash, module.clone());
        module
    }
}

impl GpuExecutor for TestGpuExecutor {
    fn execute_with_format(
        &self,
        ops: &[GpuOp],
        input: &[u8],
        width: u32,
        height: u32,
        buffer_format: rasmcore_pipeline::BufferFormat,
    ) -> Result<Vec<u8>, GpuError> {
        if ops.is_empty() {
            return Ok(input.to_vec());
        }

        let bpp = buffer_format.bytes_per_pixel() as u64;
        let buf_size = (width as u64) * (height as u64) * bpp;
        if buf_size as usize > self.max_buffer {
            return Err(GpuError::BufferTooLarge {
                requested: buf_size as usize,
                max: self.max_buffer,
            });
        }

        let buf_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-buf-a"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-buf-b"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&buf_a, 0, input);

        let mut read_buf = &buf_a;
        let mut write_buf = &buf_b;
        let mut snapshots: HashMap<u32, wgpu::Buffer> = HashMap::new();

        for (i, op) in ops.iter().enumerate() {
            match op {
                GpuOp::Snapshot { binding } => {
                    let snap = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("snap"),
                        size: buf_size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut enc = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    enc.copy_buffer_to_buffer(read_buf, 0, &snap, 0, buf_size);
                    self.queue.submit(std::iter::once(enc.finish()));
                    snapshots.insert(*binding, snap);
                }
                GpuOp::Compute {
                    shader,
                    entry_point,
                    workgroup_size,
                    params,
                    extra_buffers,
                    ..
                } => {
                    let shader_module = self.get_shader(shader);
                    let uniform_buf = if !params.is_empty() {
                        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: None,
                            size: params.len() as u64,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        self.queue.write_buffer(&buf, 0, params);
                        Some(buf)
                    } else {
                        None
                    };
                    let extra_bufs: Vec<wgpu::Buffer> = extra_buffers
                        .iter()
                        .map(|data| {
                            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: None,
                                size: data.len() as u64,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            self.queue.write_buffer(&buf, 0, data);
                            buf
                        })
                        .collect();

                    let mut next_binding = 2u32;
                    if uniform_buf.is_some() {
                        next_binding += 1;
                    }
                    let extra_start = next_binding;
                    next_binding += extra_bufs.len() as u32;

                    let mut layout_entries = vec![
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ];
                    if uniform_buf.is_some() {
                        layout_entries.push(wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        });
                    }
                    for idx in 0..extra_bufs.len() as u32 {
                        layout_entries.push(wgpu::BindGroupLayoutEntry {
                            binding: extra_start + idx,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        });
                    }
                    let mut sorted_snaps: Vec<u32> = snapshots
                        .keys()
                        .copied()
                        .filter(|b| *b >= next_binding)
                        .collect();
                    sorted_snaps.sort();
                    for &sb in &sorted_snaps {
                        layout_entries.push(wgpu::BindGroupLayoutEntry {
                            binding: sb,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        });
                    }

                    let bgl =
                        self.device
                            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                label: None,
                                entries: &layout_entries,
                            });
                    let mut bg_entries = vec![
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: read_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: write_buf.as_entire_binding(),
                        },
                    ];
                    if let Some(ref ub) = uniform_buf {
                        bg_entries.push(wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ub.as_entire_binding(),
                        });
                    }
                    for (idx, eb) in extra_bufs.iter().enumerate() {
                        bg_entries.push(wgpu::BindGroupEntry {
                            binding: extra_start + idx as u32,
                            resource: eb.as_entire_binding(),
                        });
                    }
                    for &sb in &sorted_snaps {
                        bg_entries.push(wgpu::BindGroupEntry {
                            binding: sb,
                            resource: snapshots[&sb].as_entire_binding(),
                        });
                    }
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &bgl,
                        entries: &bg_entries,
                    });
                    let pl = self
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&bgl],
                            push_constant_ranges: &[],
                        });
                    let pipeline =
                        self.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: None,
                                layout: Some(&pl),
                                module: &shader_module,
                                entry_point: Some(entry_point),
                                compilation_options: Default::default(),
                                cache: None,
                            });
                    let [wg_x, wg_y, _] = *workgroup_size;
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            (width + wg_x - 1) / wg_x,
                            (height + wg_y - 1) / wg_y,
                            1,
                        );
                    }
                    self.queue.submit(std::iter::once(encoder.finish()));
                    if i < ops.len() - 1 {
                        std::mem::swap(&mut read_buf, &mut write_buf);
                    }
                }
            }
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-readback"),
            });
        encoder.copy_buffer_to_buffer(write_buf, 0, &staging, 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionError(format!("readback recv: {e}")))?
            .map_err(|e| GpuError::ExecutionError(format!("readback map: {e}")))?;

        Ok(slice.get_mapped_range().to_vec())
    }

    fn max_buffer_size(&self) -> usize {
        self.max_buffer
    }
}

// ─── Test Helpers ──────────────────────────────────────────────────────────

fn try_gpu() -> Option<TestGpuExecutor> {
    match TestGpuExecutor::try_new() {
        Ok(exec) => {
            eprintln!("  GPU available: {}", exec.adapter_name);
            Some(exec)
        }
        Err(GpuError::NotAvailable(msg)) => {
            eprintln!("  GPU not available (skipping): {msg}");
            None
        }
        Err(e) => panic!("Unexpected GPU error: {e}"),
    }
}

fn make_gradient_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            p.push(((x * 255) / w) as u8);
            p.push(((y * 255) / h) as u8);
            p.push((((x + y) * 128) / (w + h)) as u8);
            p.push(255u8); // alpha
        }
    }
    p
}

fn info_rgba8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgba8,
        color_space: ColorSpace::Srgb,
    }
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

fn max_channel_error(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn assert_gpu_parity(label: &str, cpu: &[u8], gpu: &[u8], max_mae: f64) {
    let mae = mean_absolute_error(cpu, gpu);
    let max_err = max_channel_error(cpu, gpu);
    assert!(
        mae <= max_mae,
        "{label}: MAE={mae:.4} exceeds threshold {max_mae}. max_channel_err={max_err}"
    );
    eprintln!("  {label}: MAE={mae:.4}, max_channel_err={max_err} (threshold: MAE<={max_mae})");
}

/// Convert RGB8 pixels to RGBA8 by inserting alpha=255.
fn rgb8_to_rgba8(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.extend_from_slice(chunk);
        rgba.push(255);
    }
    rgba
}

/// Extract RGB channels from RGBA8 pixels (drop alpha).
fn rgba8_to_rgb8(rgba: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
    for chunk in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&chunk[..3]);
    }
    rgb
}

// ─── GPU vs CPU Parity Tests ──────────────────────────────────────────────

use rasmcore_image::domain::pipeline::nodes::filters::*;

#[test]
fn gpu_cpu_parity_gaussian_blur() {
    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let pixels = make_gradient_rgba(w, h);
    let info = info_rgba8(w, h);
    let rect = Rect::new(0, 0, w, h);

    // CPU path
    let cpu_output = filters::BlurParams { radius: 3.0 }
        .compute(rect, &mut |_| Ok(pixels.clone()), &info)
        .unwrap();

    // GPU path
    let node = BlurNode::new(0, info.clone(), filters::BlurParams { radius: 3.0 });
    let ops = node
        .gpu_ops(w, h)
        .expect("blur should support GPU for RGBA8");
    let gpu_output = gpu
        .execute_with_format(
            &ops,
            &pixels,
            w,
            h,
            rasmcore_pipeline::gpu::BufferFormat::U32Packed,
        )
        .unwrap();

    // Tolerance accounts for edge-handling differences: CPU expands+crops
    // with border reflection, GPU clamps coordinates at image bounds.
    assert_gpu_parity("gaussian_blur r=3.0", &cpu_output, &gpu_output, 2.0);
}

#[test]
fn gpu_cpu_parity_bilateral() {
    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let info_rgb = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };
    let info_rgba = info_rgba8(w, h);
    let rect = Rect::new(0, 0, w, h);

    // Generate RGB8 gradient for CPU (bilateral requires Gray8/RGB8)
    let mut rgb_pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            rgb_pixels.push(((x * 255) / w) as u8);
            rgb_pixels.push(((y * 255) / h) as u8);
            rgb_pixels.push((((x + y) * 128) / (w + h)) as u8);
        }
    }
    // RGBA8 version for GPU (same pixels, alpha=255)
    let rgba_pixels = rgb8_to_rgba8(&rgb_pixels);

    let config = filters::BilateralParams {
        diameter: 5,
        sigma_color: 75.0,
        sigma_space: 75.0,
    };

    // CPU path (RGB8)
    let cpu_rgb = config
        .compute(rect, &mut |_| Ok(rgb_pixels.clone()), &info_rgb)
        .unwrap();

    // GPU path (RGBA8)
    let node = BilateralNode::new(0, info_rgba.clone(), config.clone());
    let ops = node
        .gpu_ops(w, h)
        .expect("bilateral should support GPU for RGBA8");
    let gpu_rgba = gpu
        .execute_with_format(
            &ops,
            &rgba_pixels,
            w,
            h,
            rasmcore_pipeline::gpu::BufferFormat::U32Packed,
        )
        .unwrap();

    // Compare RGB channels only (GPU adds alpha passthrough)
    let gpu_rgb = rgba8_to_rgb8(&gpu_rgba);
    assert_gpu_parity("bilateral d=5 sc=75 ss=75", &cpu_rgb, &gpu_rgb, 2.0);
}

/// Spherize parity test.
///
/// The CPU uses powf-based distortion with EWA sampling, while the GPU uses
/// asin-based distortion with bilinear sampling. These are intentionally
/// different algorithms optimized for their respective targets. We validate
/// that both produce a visually similar spherize effect with loose tolerance.
///
/// For a small amount (0.15), the formulas converge more closely.
#[test]
fn gpu_cpu_parity_spherize() {
    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let pixels = make_gradient_rgba(w, h);
    let info = info_rgba8(w, h);
    let rect = Rect::new(0, 0, w, h);

    // Use small amount where the two algorithms converge more closely
    let config = filters::SpherizeParams { amount: 0.15 };

    // CPU path
    let cpu_output = config
        .compute(rect, &mut |_| Ok(pixels.clone()), &info)
        .unwrap();

    // GPU path
    let node = SpherizeNode::new(0, info.clone(), config.clone());
    let ops = node
        .gpu_ops(w, h)
        .expect("spherize should support GPU for RGBA8");
    let gpu_output = gpu
        .execute_with_format(
            &ops,
            &pixels,
            w,
            h,
            rasmcore_pipeline::gpu::BufferFormat::U32Packed,
        )
        .unwrap();

    // Wider tolerance: different distortion formulas (powf vs asin) and
    // different sampling (EWA vs bilinear) produce visually similar but
    // numerically different results.
    assert_gpu_parity("spherize amount=0.15", &cpu_output, &gpu_output, 3.0);
}

#[test]
fn gpu_cpu_parity_lut3d() {
    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let pixels = make_gradient_rgba(w, h);
    let info = info_rgba8(w, h);

    // Build a non-trivial 3D LUT: warm color grading (boost reds, reduce blues)
    let grid_size = 17;
    let mut lut = ColorLut3D::identity(grid_size);
    let scale = 1.0 / (grid_size - 1) as f32;
    for b in 0..grid_size {
        for g in 0..grid_size {
            for r in 0..grid_size {
                let idx = b * grid_size * grid_size + g * grid_size + r;
                let rv = r as f32 * scale;
                let bv = b as f32 * scale;
                // Warm shift: boost red, reduce blue
                lut.data[idx][0] = (rv * 1.1).min(1.0);
                lut.data[idx][2] = bv * 0.9;
            }
        }
    }

    // CPU path: apply the LUT to RGBA8 pixels
    let cpu_output = lut.apply(&pixels, &info).unwrap();

    // GPU path: construct GpuOps manually (mirrors FusedClutNode::gpu_ops)
    let grid = grid_size as u32;
    let mut params = Vec::with_capacity(16);
    params.extend_from_slice(&w.to_le_bytes());
    params.extend_from_slice(&h.to_le_bytes());
    params.extend_from_slice(&grid.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes()); // padding

    let mut lut_buf = Vec::with_capacity(lut.data.len() * 16);
    for entry in &lut.data {
        lut_buf.extend_from_slice(&entry[0].to_le_bytes());
        lut_buf.extend_from_slice(&entry[1].to_le_bytes());
        lut_buf.extend_from_slice(&entry[2].to_le_bytes());
        lut_buf.extend_from_slice(&0.0f32.to_le_bytes());
    }

    let lut3d_shader = rasmcore_gpu_shaders::with_pixel_ops(include_str!(
        "../src/domain/pipeline/shaders/lut3d.wgsl"
    ));
    let ops = vec![GpuOp::Compute {
        shader: lut3d_shader,
        entry_point: "main",
        workgroup_size: [256, 1, 1],
        params,
        extra_buffers: vec![lut_buf],
        buffer_format: Default::default(),
    }];

    let gpu_output = gpu
        .execute_with_format(
            &ops,
            &pixels,
            w,
            h,
            rasmcore_pipeline::gpu::BufferFormat::U32Packed,
        )
        .unwrap();

    assert_gpu_parity("lut3d warm_grade grid=17", &cpu_output, &gpu_output, 1.0);
}

// ─── f32 Buffer Format Parity Tests ────────────────────────────────────────

/// Convert RGBA8 bytes to RGBA32f bytes (0–255 mapped to 0.0–255.0, like GPU unpack).
fn rgba8_to_f32_bytes(rgba8: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba8.len() * 4);
    for &byte in rgba8 {
        out.extend_from_slice(&(byte as f32).to_le_bytes());
    }
    out
}

/// Convert RGBA32f bytes back to RGBA8 bytes (clamp + round).
fn f32_bytes_to_rgba8(f32_bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f32_bytes.len() / 4);
    for chunk in f32_bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(val.round().clamp(0.0, 255.0) as u8);
    }
    out
}

#[test]
fn gpu_f32_parity_gaussian_blur() {
    use rasmcore_image::domain::filter_traits::GpuFilter;
    use rasmcore_pipeline::gpu::BufferFormat;

    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let pixels_u8 = make_gradient_rgba(w, h);
    let pixels_f32 = rgba8_to_f32_bytes(&pixels_u8);
    let info = info_rgba8(w, h);
    let rect = Rect::new(0, 0, w, h);

    let config = filters::BlurParams { radius: 3.0 };

    // CPU reference (u8 domain)
    let cpu_output_u8 = config
        .compute(rect, &mut |_| Ok(pixels_u8.clone()), &info)
        .unwrap();

    // GPU u32-packed (baseline)
    let ops_u32 = config
        .gpu_ops_with_format(w, h, BufferFormat::U32Packed)
        .expect("blur should support u32 GPU");
    let gpu_u32 = gpu
        .execute_with_format(&ops_u32, &pixels_u8, w, h, BufferFormat::U32Packed)
        .unwrap();

    // GPU f32 vec4
    let ops_f32 = config
        .gpu_ops_with_format(w, h, BufferFormat::F32Vec4)
        .expect("blur should support f32 GPU");
    let gpu_f32_bytes = gpu
        .execute_with_format(&ops_f32, &pixels_f32, w, h, BufferFormat::F32Vec4)
        .unwrap();

    // Convert f32 GPU output back to u8 for comparison
    let gpu_f32_as_u8 = f32_bytes_to_rgba8(&gpu_f32_bytes);

    // Both GPU paths should match CPU output closely
    assert_gpu_parity("blur_f32 vs cpu", &cpu_output_u8, &gpu_f32_as_u8, 2.0);
    assert_gpu_parity("blur_u32 vs cpu", &cpu_output_u8, &gpu_u32, 2.0);

    // f32 GPU vs u32 GPU: the f32 path preserves more precision across H+V passes.
    // u32 packs to 8-bit between passes, losing ~0.4 bits/pass. This difference
    // IS the precision improvement f32 provides. Allow up to 2.0 MAE.
    assert_gpu_parity("blur_f32 vs u32", &gpu_u32, &gpu_f32_as_u8, 2.0);

    eprintln!("  f32 blur parity: PASS");
}

#[test]
fn gpu_f32_parity_affine_resample() {
    use rasmcore_pipeline::gpu::BufferFormat;

    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let pixels_u8 = make_gradient_rgba(w, h);
    let pixels_f32 = rgba8_to_f32_bytes(&pixels_u8);
    let info_u8 = info_rgba8(w, h);
    let info_f32 = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgba32f,
        color_space: ColorSpace::Srgb,
    };

    // 2x downscale matrix: x' = 0.5*x, y' = 0.5*y
    let matrix = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0];
    let (out_w, out_h) = (32, 32);

    // GPU u32
    // Note: the graph walker passes OUTPUT dimensions to the executor
    // (the node's info() returns output dims). Input pixels cover the full source.
    let node_u32 = rasmcore_image::domain::pipeline::nodes::transform::ComposedAffineNode::new(
        0, info_u8, matrix, out_w, out_h,
    );
    let ops_u32 = node_u32
        .gpu_ops_with_format(out_w, out_h, BufferFormat::U32Packed)
        .expect("affine should support u32 GPU");
    // Pad input to output-sized buffer (executor expects width*height*bpp input)
    let mut input_u32_padded = vec![0u8; (out_w * out_h * 4) as usize];
    // Copy source rows into padded buffer (source is larger, so just fill what fits)
    for y in 0..out_h.min(h) {
        let src_off = (y * w * 4) as usize;
        let dst_off = (y * out_w * 4) as usize;
        let row_bytes = (out_w.min(w) * 4) as usize;
        input_u32_padded[dst_off..dst_off + row_bytes]
            .copy_from_slice(&pixels_u8[src_off..src_off + row_bytes]);
    }
    let gpu_u32 = gpu
        .execute_with_format(
            &ops_u32,
            &input_u32_padded,
            out_w,
            out_h,
            BufferFormat::U32Packed,
        )
        .unwrap();

    // GPU f32
    let node_f32 = rasmcore_image::domain::pipeline::nodes::transform::ComposedAffineNode::new(
        0, info_f32, matrix, out_w, out_h,
    );
    let ops_f32 = node_f32
        .gpu_ops_with_format(out_w, out_h, BufferFormat::F32Vec4)
        .expect("affine should support f32 GPU");
    let mut input_f32_padded = vec![0u8; (out_w * out_h * 16) as usize];
    for y in 0..out_h.min(h) {
        let src_off = (y * w * 16) as usize;
        let dst_off = (y * out_w * 16) as usize;
        let row_bytes = (out_w.min(w) * 16) as usize;
        input_f32_padded[dst_off..dst_off + row_bytes]
            .copy_from_slice(&pixels_f32[src_off..src_off + row_bytes]);
    }
    let gpu_f32_bytes = gpu
        .execute_with_format(
            &ops_f32,
            &input_f32_padded,
            out_w,
            out_h,
            BufferFormat::F32Vec4,
        )
        .unwrap();

    // Convert f32 output to u8
    let gpu_f32_as_u8 = f32_bytes_to_rgba8(&gpu_f32_bytes);

    // Verify output sizes
    assert_eq!(
        gpu_f32_bytes.len(),
        (out_w * out_h * 16) as usize,
        "f32 output size"
    );
    assert_eq!(
        gpu_u32.len(),
        (out_w * out_h * 4) as usize,
        "u32 output size"
    );

    // Both should produce similar results
    assert_gpu_parity("affine_f32 vs u32", &gpu_u32, &gpu_f32_as_u8, 1.0);

    eprintln!("  f32 affine resample parity: PASS");
}

// ─── f32 Point Op Parity Tests ─────────────────────────────────────────────

#[test]
fn gpu_f32_gaussian_noise_deterministic() {
    use rasmcore_image::domain::filter_traits::GpuFilter;
    use rasmcore_image::domain::filters::GaussianNoiseParams;
    use rasmcore_pipeline::gpu::BufferFormat;

    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let pixels_u8 = make_gradient_rgba(w, h);
    let pixels_f32_norm: Vec<u8> = pixels_u8
        .iter()
        .flat_map(|&b| (b as f32 / 255.0).to_le_bytes())
        .collect();

    let config = GaussianNoiseParams {
        amount: 50.0,
        mean: 0.0,
        sigma: 25.0,
        seed: 12345,
    };

    let ops = config
        .gpu_ops_with_format(w, h, BufferFormat::F32Vec4)
        .expect("gaussian_noise should support f32 GPU");

    // Run twice — same seed should produce identical output
    let out1 = gpu
        .execute_with_format(&ops, &pixels_f32_norm, w, h, BufferFormat::F32Vec4)
        .unwrap();
    let out2 = gpu
        .execute_with_format(&ops, &pixels_f32_norm, w, h, BufferFormat::F32Vec4)
        .unwrap();
    assert_eq!(out1, out2, "same seed must produce identical output");

    // Output should differ from input (noise was applied)
    assert_ne!(out1, pixels_f32_norm, "noise should change the image");

    // Verify values are in valid range [0, 1]
    for chunk in out1.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(v >= 0.0 && v <= 1.0, "pixel value {v} out of [0,1] range");
    }

    eprintln!("  gaussian_noise f32 GPU: deterministic + valid range: PASS");
}

#[test]
fn gpu_f32_uniform_noise_deterministic() {
    use rasmcore_image::domain::filter_traits::GpuFilter;
    use rasmcore_image::domain::filters::UniformNoiseParams;
    use rasmcore_pipeline::gpu::BufferFormat;

    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let pixels_u8 = make_gradient_rgba(w, h);
    let pixels_f32_norm: Vec<u8> = pixels_u8
        .iter()
        .flat_map(|&b| (b as f32 / 255.0).to_le_bytes())
        .collect();

    let config = UniformNoiseParams {
        range: 40.0,
        seed: 99999,
    };

    let ops = config
        .gpu_ops_with_format(w, h, BufferFormat::F32Vec4)
        .expect("uniform_noise should support f32 GPU");

    let out1 = gpu
        .execute_with_format(&ops, &pixels_f32_norm, w, h, BufferFormat::F32Vec4)
        .unwrap();
    let out2 = gpu
        .execute_with_format(&ops, &pixels_f32_norm, w, h, BufferFormat::F32Vec4)
        .unwrap();
    assert_eq!(out1, out2, "same seed must produce identical output");
    assert_ne!(out1, pixels_f32_norm, "noise should change the image");

    for chunk in out1.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(v >= 0.0 && v <= 1.0, "pixel value {v} out of [0,1] range");
    }

    eprintln!("  uniform_noise f32 GPU: deterministic + valid range: PASS");
}

#[test]
fn gpu_f32_film_grain_deterministic() {
    use rasmcore_image::domain::filter_traits::GpuFilter;
    use rasmcore_image::domain::filters::FilmGrainParams;
    use rasmcore_pipeline::gpu::BufferFormat;

    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64, 64);
    let pixels_u8 = make_gradient_rgba(w, h);
    let pixels_f32_norm: Vec<u8> = pixels_u8
        .iter()
        .flat_map(|&b| (b as f32 / 255.0).to_le_bytes())
        .collect();

    let config = FilmGrainParams {
        amount: 0.5,
        size: 2.0,
        seed: 42,
    };

    let ops = config
        .gpu_ops_with_format(w, h, BufferFormat::F32Vec4)
        .expect("film_grain should support f32 GPU");

    let out1 = gpu
        .execute_with_format(&ops, &pixels_f32_norm, w, h, BufferFormat::F32Vec4)
        .unwrap();
    let out2 = gpu
        .execute_with_format(&ops, &pixels_f32_norm, w, h, BufferFormat::F32Vec4)
        .unwrap();
    assert_eq!(out1, out2, "same seed must produce identical output");
    assert_ne!(out1, pixels_f32_norm, "film grain should change the image");

    for chunk in out1.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(v >= 0.0 && v <= 1.0, "pixel value {v} out of [0,1] range");
    }

    eprintln!("  film_grain f32 GPU: deterministic + valid range: PASS");
}

// ─── Fusion Node GPU Tests ─────────────────────────────────────────────────

#[test]
fn gpu_f32_fused_lut_brightness() {
    use rasmcore_pipeline::gpu::BufferFormat;

    let gpu = match try_gpu() {
        Some(g) => g,
        None => return,
    };

    let (w, h) = (64u32, 64u32);
    let pixels_f32: Vec<u8> = (0..(w * h) as usize)
        .flat_map(|i| {
            let r = (i % w as usize) as f32 / w as f32;
            let g = (i / w as usize) as f32 / h as f32;
            [r, g, 0.5f32, 1.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect();

    // Build LUT for brightness +0.2
    let lut = rasmcore_image::domain::point_ops::build_lut(
        &rasmcore_image::domain::point_ops::PointOp::Brightness(0.2),
    );

    // Pack LUT as u32 array (same as FusedLutNode does)
    let mut lut_buf = Vec::with_capacity(256 * 4);
    for &v in lut.iter() {
        lut_buf.extend_from_slice(&(v as u32).to_le_bytes());
    }

    let mut params = Vec::with_capacity(8);
    params.extend_from_slice(&w.to_le_bytes());
    params.extend_from_slice(&h.to_le_bytes());

    // Use the f32 LUT shader directly
    let shader = include_str!("../src/domain/pipeline/shaders/lut1d_f32.wgsl").to_string();

    let ops = vec![GpuOp::Compute {
        shader,
        entry_point: "main",
        workgroup_size: [16, 16, 1],
        params,
        extra_buffers: vec![lut_buf.clone()],
        buffer_format: BufferFormat::F32Vec4,
    }];

    let gpu_output = gpu
        .execute_with_format(&ops, &pixels_f32, w, h, BufferFormat::F32Vec4)
        .unwrap();

    // CPU f32 reference — interpolated LUT lookup
    let cpu_output: Vec<u8> = pixels_f32
        .chunks_exact(16)
        .flat_map(|px| {
            let r = f32::from_le_bytes([px[0], px[1], px[2], px[3]]);
            let g = f32::from_le_bytes([px[4], px[5], px[6], px[7]]);
            let b = f32::from_le_bytes([px[8], px[9], px[10], px[11]]);
            let a = f32::from_le_bytes([px[12], px[13], px[14], px[15]]);

            let apply = |v: f32| -> f32 {
                let s = (v * 255.0).clamp(0.0, 255.0);
                let lo = s.floor() as usize;
                let hi = (lo + 1).min(255);
                let frac = s - lo as f32;
                (lut[lo] as f32 * (1.0 - frac) + lut[hi] as f32 * frac) / 255.0
            };
            [apply(r), apply(g), apply(b), a]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect();

    // Compare f32 outputs
    let max_diff: f32 = cpu_output
        .chunks_exact(4)
        .zip(gpu_output.chunks_exact(4))
        .map(|(a, b)| {
            let va = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
            let vb = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            (va - vb).abs()
        })
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 0.005,
        "FusedLut f32: max_diff={max_diff} (threshold 0.005)"
    );
    eprintln!("  fused_lut_1d f32 GPU: max_diff={max_diff:.6} — PASS");
}
