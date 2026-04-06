//! V2 GPU executor — wgpu-based implementation of the V2 GpuExecutor trait.
//!
//! Takes `&[GpuShader]` (V2 node.rs types) and returns `Vec<f32>`.
//! Shares the same wgpu device/buffer pooling pattern as the V1 executor.

pub mod display;

use rasmcore_pipeline_v2::gpu::{GpuError, GpuExecutor};
use rasmcore_pipeline_v2::node::GpuShader;
use std::collections::HashMap;

/// FNV-1a 64-bit hash for shader and buffer content caching.
fn content_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

struct CachedBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

/// V2 GPU executor with buffer pooling.
pub struct WgpuExecutorV2 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_cache: std::cell::RefCell<HashMap<u64, wgpu::ShaderModule>>,
    adapter_name: String,
    max_buffer: usize,
    ping_pong: std::cell::RefCell<Option<(CachedBuffer, CachedBuffer, u64)>>,
    staging: std::cell::RefCell<Option<CachedBuffer>>,
    uniform: std::cell::RefCell<Option<CachedBuffer>>,
    extra_cache: std::cell::RefCell<HashMap<u64, wgpu::Buffer>>,
}

impl WgpuExecutorV2 {
    /// Try to create a V2 GPU executor. Returns Err if no GPU is available.
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
                label: Some("rasmcore-v2-gpu"),
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
            ping_pong: std::cell::RefCell::new(None),
            staging: std::cell::RefCell::new(None),
            uniform: std::cell::RefCell::new(None),
            extra_cache: std::cell::RefCell::new(HashMap::new()),
        })
    }

    /// GPU adapter name (e.g., "Apple M1 Pro").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Access the wgpu device (for display target creation).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the wgpu queue (for display target submission).
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn get_shader(&self, source: &str) -> Result<wgpu::ShaderModule, GpuError> {
        let hash = content_hash(source.as_bytes());
        let mut cache = self.shader_cache.borrow_mut();
        if let Some(module) = cache.get(&hash) {
            return Ok(module.clone());
        }

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rasmcore-v2-compute"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        cache.insert(hash, module.clone());
        Ok(module)
    }

    fn ensure_ping_pong(&self, buf_size: u64) -> (wgpu::Buffer, wgpu::Buffer) {
        let mut pp = self.ping_pong.borrow_mut();
        if let Some((ref a, ref b, cached_size)) = *pp {
            if cached_size >= buf_size {
                return (a.buffer.clone(), b.buffer.clone());
            }
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buf_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-buf-a"),
            size: buf_size,
            usage,
            mapped_at_creation: false,
        });
        let buf_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-buf-b"),
            size: buf_size,
            usage,
            mapped_at_creation: false,
        });
        *pp = Some((
            CachedBuffer { buffer: buf_a.clone(), size: buf_size },
            CachedBuffer { buffer: buf_b.clone(), size: buf_size },
            buf_size,
        ));
        (buf_a, buf_b)
    }

    fn ensure_staging(&self, buf_size: u64) -> wgpu::Buffer {
        let mut s = self.staging.borrow_mut();
        if let Some(ref c) = *s {
            if c.size >= buf_size {
                return c.buffer.clone();
            }
        }
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *s = Some(CachedBuffer { buffer: buf.clone(), size: buf_size });
        buf
    }

    fn ensure_uniform(&self, needed: u64) -> wgpu::Buffer {
        let mut u = self.uniform.borrow_mut();
        if let Some(ref c) = *u {
            if c.size >= needed {
                return c.buffer.clone();
            }
        }
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-uniform"),
            size: needed,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *u = Some(CachedBuffer { buffer: buf.clone(), size: needed });
        buf
    }

    fn get_or_create_extra(&self, data: &[u8]) -> wgpu::Buffer {
        let hash = content_hash(data);
        let mut cache = self.extra_cache.borrow_mut();
        if let Some(buf) = cache.get(&hash) {
            return buf.clone();
        }
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-extra"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, data);
        cache.insert(hash, buf.clone());
        buf
    }
}

impl GpuExecutor for WgpuExecutorV2 {
    fn prepare(&self, shader_sources: &[String]) {
        for source in shader_sources {
            // get_shader() compiles and caches by content hash
            let _ = self.get_shader(source);
        }
    }

    fn execute(
        &self,
        ops: &[GpuShader],
        input: &[f32],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, GpuError> {
        if ops.is_empty() {
            return Ok(input.to_vec());
        }

        // f32 RGBA: 4 floats * 4 bytes = 16 bytes per pixel
        let buf_size = (width as u64) * (height as u64) * 16;
        if buf_size as usize > self.max_buffer {
            return Err(GpuError::BufferTooLarge {
                requested: buf_size as usize,
                max: self.max_buffer,
            });
        }

        // Pre-scan: max uniform size across all ops
        let max_uniform = ops
            .iter()
            .filter(|op| !op.params.is_empty())
            .map(|op| op.params.len() as u64)
            .max()
            .unwrap_or(0);

        let (buf_a, buf_b) = self.ensure_ping_pong(buf_size);
        let uniform_buf = if max_uniform > 0 {
            // Ensure 16-byte aligned minimum
            let aligned = ((max_uniform + 15) / 16) * 16;
            Some(self.ensure_uniform(aligned))
        } else {
            None
        };

        // Upload input f32 data
        let input_bytes: &[u8] = bytemuck::cast_slice(input);
        self.queue.write_buffer(&buf_a, 0, input_bytes);

        let mut read_buf = &buf_a;
        let mut write_buf = &buf_b;

        for (i, op) in ops.iter().enumerate() {
            let shader_module = self.get_shader(&op.body)?;

            // Write uniform params
            if !op.params.is_empty() {
                if let Some(ref ub) = uniform_buf {
                    self.queue.write_buffer(ub, 0, &op.params);
                }
            }

            // Extra buffers
            let extra_bufs: Vec<wgpu::Buffer> = op
                .extra_buffers
                .iter()
                .map(|data| self.get_or_create_extra(data))
                .collect();

            // Bind group layout: 0=input, 1=output, 2=uniform (optional), 3+=extras
            let mut next_binding = 2u32;
            let has_uniform = uniform_buf.is_some() && !op.params.is_empty();
            if has_uniform {
                next_binding += 1;
            }
            let extra_start = next_binding;

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

            if has_uniform {
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

            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("v2-layout"),
                        entries: &layout_entries,
                    });

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

            if has_uniform {
                if let Some(ref ub) = uniform_buf {
                    bg_entries.push(wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ub.as_entire_binding(),
                    });
                }
            }

            for (idx, eb) in extra_bufs.iter().enumerate() {
                bg_entries.push(wgpu::BindGroupEntry {
                    binding: extra_start + idx as u32,
                    resource: eb.as_entire_binding(),
                });
            }

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("v2-bind"),
                layout: &bind_group_layout,
                entries: &bg_entries,
            });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("v2-pipeline-layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("v2-pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: Some(op.entry_point),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            let [wg_x, wg_y, _wg_z] = op.workgroup_size;
            let dispatch_x = (width + wg_x - 1) / wg_x;
            let dispatch_y = (height + wg_y - 1) / wg_y;

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("v2-encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("v2-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));

            // Ping-pong: swap for next op
            if i < ops.len() - 1 {
                std::mem::swap(&mut read_buf, &mut write_buf);
            }
        }

        // Readback from write_buf
        let staging = self.ensure_staging(buf_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("v2-readback"),
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

        let floats: Vec<f32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        staging.unmap();
        Ok(floats)
    }

    fn max_buffer_size(&self) -> usize {
        self.max_buffer
    }
}
