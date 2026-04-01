//! GPU executor — wgpu-based implementation of GpuExecutor trait.
//!
//! Auto-detects GPU on startup, compiles WGSL shaders, manages buffers,
//! and executes compute dispatches via the best available backend
//! (Metal on macOS, Vulkan on Linux, DX12 on Windows).

use rasmcore_pipeline::gpu::{GpuError, GpuExecutor, GpuOp};
use std::collections::HashMap;
use wgpu;

/// Hash a shader source string for caching.
fn shader_hash(source: &str) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in source.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Native GPU executor wrapping wgpu device and shader cache.
pub struct WgpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_cache: std::cell::RefCell<HashMap<u64, wgpu::ShaderModule>>,
    adapter_name: String,
    max_buffer: usize,
}

impl WgpuExecutor {
    /// Try to create a GPU executor. Returns None if no GPU is available.
    pub fn try_new() -> Result<Self, GpuError> {
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
                label: Some("rasmcore-gpu"),
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

    /// GPU adapter name (e.g., "Apple M1 Pro", "NVIDIA GeForce RTX 4090").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Get or compile a shader module, cached by source hash.
    fn get_shader(&self, source: &str) -> Result<wgpu::ShaderModule, GpuError> {
        let hash = shader_hash(source);
        let mut cache = self.shader_cache.borrow_mut();
        if let Some(module) = cache.get(&hash) {
            return Ok(module.clone());
        }

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rasmcore-compute"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        cache.insert(hash, module.clone());
        Ok(module)
    }
}

impl GpuExecutor for WgpuExecutor {
    fn execute(
        &self,
        ops: &[GpuOp],
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, GpuError> {
        if ops.is_empty() {
            return Ok(input.to_vec());
        }

        let buf_size = (width * height * 4) as u64; // RGBA8
        if buf_size as usize > self.max_buffer {
            return Err(GpuError::BufferTooLarge {
                requested: buf_size as usize,
                max: self.max_buffer,
            });
        }

        // Create ping-pong storage buffers
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

        // Upload input to buf_a
        self.queue.write_buffer(&buf_a, 0, input);

        let mut read_buf = &buf_a;
        let mut write_buf = &buf_b;

        // Execute each op
        for (i, op) in ops.iter().enumerate() {
            let shader = self.get_shader(op.shader)?;

            // Create uniform buffer for params
            let uniform_buf = if !op.params.is_empty() {
                let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpu-uniform"),
                    size: op.params.len() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(&buf, 0, &op.params);
                Some(buf)
            } else {
                None
            };

            // Create extra storage buffers
            let extra_bufs: Vec<wgpu::Buffer> = op
                .extra_buffers
                .iter()
                .map(|data| {
                    let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("gpu-extra"),
                        size: data.len() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.queue.write_buffer(&buf, 0, data);
                    buf
                })
                .collect();

            // Build bind group layout entries
            let mut layout_entries = vec![
                // binding 0: input (read)
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
                // binding 1: output (write)
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

            let mut next_binding = 2u32;

            // binding 2: uniform params (if present)
            if uniform_buf.is_some() {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: next_binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                next_binding += 1;
            }

            // Extra storage buffers
            for _ in &extra_bufs {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: next_binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                next_binding += 1;
            }

            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("gpu-layout"),
                        entries: &layout_entries,
                    });

            // Build bind group entries
            let mut bg_entries: Vec<wgpu::BindGroupEntry> = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: read_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: write_buf.as_entire_binding(),
                },
            ];

            let mut binding_idx = 2u32;
            if let Some(ref ub) = uniform_buf {
                bg_entries.push(wgpu::BindGroupEntry {
                    binding: binding_idx,
                    resource: ub.as_entire_binding(),
                });
                binding_idx += 1;
            }
            for eb in &extra_bufs {
                bg_entries.push(wgpu::BindGroupEntry {
                    binding: binding_idx,
                    resource: eb.as_entire_binding(),
                });
                binding_idx += 1;
            }

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu-bind"),
                layout: &bind_group_layout,
                entries: &bg_entries,
            });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("gpu-pipeline-layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("gpu-pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: Some(op.entry_point),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            // Dispatch
            let [wg_x, wg_y, _wg_z] = op.workgroup_size;
            let dispatch_x = (width + wg_x - 1) / wg_x;
            let dispatch_y = (height + wg_y - 1) / wg_y;

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpu-encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));

            // Ping-pong: swap read/write for next op
            if i < ops.len() - 1 {
                std::mem::swap(&mut read_buf, &mut write_buf);
            }
        }

        // Readback from write_buf (the last write destination)
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

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionError(format!("readback recv: {e}")))?
            .map_err(|e| GpuError::ExecutionError(format!("readback map: {e}")))?;

        let data = slice.get_mapped_range().to_vec();
        Ok(data)
    }

    fn max_buffer_size(&self) -> usize {
        self.max_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_executor_creation() {
        // This test will pass on machines with a GPU and fail gracefully on CI
        match WgpuExecutor::try_new() {
            Ok(exec) => {
                eprintln!("GPU available: {}", exec.adapter_name());
                assert!(!exec.adapter_name().is_empty());
                assert!(exec.max_buffer_size() > 0);
            }
            Err(GpuError::NotAvailable(msg)) => {
                eprintln!("No GPU available (expected in CI): {msg}");
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn gpu_passthrough_empty_ops() {
        if let Ok(exec) = WgpuExecutor::try_new() {
            let input = vec![128u8; 4 * 4 * 4]; // 4x4 RGBA
            let output = exec.execute(&[], &input, 4, 4).unwrap();
            assert_eq!(input, output);
        }
    }
}
