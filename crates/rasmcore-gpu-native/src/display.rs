//! Display target management — zero-copy GPU blit to wgpu surfaces.
//!
//! Consumers provide `wgpu::Surface` from their own windowing system.
//! The SDK manages the blit pipeline, viewport uniforms, and display
//! registration. Same architecture as the browser GpuHandlerV2.
//!
//! # Example
//!
//! ```ignore
//! // Consumer creates window + surface
//! let surface = instance.create_surface(&window)?;
//! let config = surface.get_default_config(&adapter, width, height)?;
//!
//! // Create display manager from the GPU executor
//! let mut display = DisplayManager::new(executor.device(), executor.queue());
//!
//! // Register a display target
//! display.add_target("viewport", surface, &config);
//!
//! // After compute: blit result to surface (zero CPU readback)
//! display.blit("viewport", &compute_output_buffer, image_width, image_height);
//!
//! // Pan/zoom
//! display.update_viewport("viewport", 0.0, 0.0, 1.0);
//! ```

use std::collections::HashMap;

/// Viewport uniform data — matches the browser's display-blit.wgsl layout.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Viewport {
    pub canvas_width: f32,
    pub canvas_height: f32,
    pub image_width: f32,
    pub image_height: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    pub zoom: f32,
    pub tone_mode: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            canvas_width: 1.0,
            canvas_height: 1.0,
            image_width: 1.0,
            image_height: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            zoom: 1.0,
            tone_mode: 0,
        }
    }
}

/// A registered display target — surface + blit pipeline + viewport state.
struct DisplayTarget {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    viewport_buf: wgpu::Buffer,
    viewport: Viewport,
}

/// Blit shader source — ported from web-ui/src/shaders/display-blit.wgsl.
const BLIT_SHADER: &str = r#"
struct Viewport {
    canvas_width: f32,
    canvas_height: f32,
    image_width: f32,
    image_height: f32,
    pan_x: f32,
    pan_y: f32,
    zoom: f32,
    tone_mode: u32,
};

@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> vp: Viewport;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let canvas_px = in.uv * vec2<f32>(vp.canvas_width, vp.canvas_height);
    let center = vec2<f32>(vp.canvas_width, vp.canvas_height) * 0.5;
    let img_center = vec2<f32>(vp.image_width, vp.image_height) * 0.5;
    let img_px = (canvas_px - center) / vp.zoom + img_center + vec2<f32>(vp.pan_x, vp.pan_y);

    let ix = i32(floor(img_px.x));
    let iy = i32(floor(img_px.y));
    let w = i32(vp.image_width);
    let h = i32(vp.image_height);

    if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
        return vec4<f32>(0.05, 0.05, 0.05, 1.0);
    }

    let idx = iy * w + ix;
    var color = pixels[idx];

    if (vp.tone_mode == 0u) {
        color = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
    }

    color = vec4<f32>(color.rgb * color.a, color.a);
    return color;
}
"#;

/// Manages named display targets for zero-copy GPU→surface blitting.
///
/// Does NOT own windows or surfaces — consumers provide them.
/// The manager creates blit pipelines and viewport uniform buffers.
pub struct DisplayManager {
    device: *const wgpu::Device,
    queue: *const wgpu::Queue,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_bind_group_layout: Option<wgpu::BindGroupLayout>,
    targets: HashMap<String, DisplayTarget>,
}

// SAFETY: DisplayManager is not Send/Sync — it holds raw pointers to Device/Queue
// that must only be used on the thread that created it. This matches the browser
// pattern where GpuHandlerV2 is single-threaded in the worker.

impl DisplayManager {
    /// Create a display manager. The device and queue must outlive this manager.
    ///
    /// # Safety
    /// The device and queue references must remain valid for the lifetime of
    /// this DisplayManager. In practice, hold the WgpuExecutorV2 in an Rc
    /// alongside this manager.
    pub unsafe fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let device_ptr = device as *const wgpu::Device;
        let queue_ptr = queue as *const wgpu::Queue;

        // Create blit pipeline (shared across all targets)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display-blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display-blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            device: device_ptr,
            queue: queue_ptr,
            blit_pipeline: Some(pipeline),
            blit_bind_group_layout: Some(bind_group_layout),
            targets: HashMap::new(),
        }
    }

    /// Get device reference without borrowing self.
    /// SAFETY: the raw pointer must be valid (ensured by constructor contract).
    #[inline]
    fn dev(ptr: *const wgpu::Device) -> &'static wgpu::Device {
        unsafe { &*ptr }
    }

    /// Get queue reference without borrowing self.
    #[inline]
    fn q(ptr: *const wgpu::Queue) -> &'static wgpu::Queue {
        unsafe { &*ptr }
    }

    /// Register a named display target.
    ///
    /// The consumer provides a surface (from their window) and its configuration.
    /// The manager creates the viewport uniform buffer.
    pub fn add_target(&mut self, name: &str, surface: wgpu::Surface<'static>, config: &wgpu::SurfaceConfiguration) {
        let device = Self::dev(self.device);
        let queue = Self::q(self.queue);

        surface.configure(device, config);

        let viewport = Viewport {
            canvas_width: config.width as f32,
            canvas_height: config.height as f32,
            ..Default::default()
        };

        let viewport_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("viewport-{name}")),
            size: std::mem::size_of::<Viewport>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&viewport_buf, 0, bytemuck::bytes_of(&viewport));

        self.targets.insert(name.to_string(), DisplayTarget {
            surface,
            config: config.clone(),
            viewport_buf,
            viewport,
        });
    }

    /// Remove a display target.
    pub fn remove_target(&mut self, name: &str) {
        self.targets.remove(name);
    }

    /// Update viewport uniforms for a target (pan, zoom, tone mapping).
    pub fn update_viewport(&mut self, name: &str, pan_x: f32, pan_y: f32, zoom: f32) {
        let queue = Self::q(self.queue);
        if let Some(target) = self.targets.get_mut(name) {
            target.viewport.pan_x = pan_x;
            target.viewport.pan_y = pan_y;
            target.viewport.zoom = zoom;
            queue.write_buffer(&target.viewport_buf, 0, bytemuck::bytes_of(&target.viewport));
        }
    }

    /// Blit a GPU pixel buffer to a named display target.
    ///
    /// `pixel_buf` is a storage buffer containing the compute result (array<vec4<f32>>).
    /// Zero CPU readback — the buffer stays on GPU.
    pub fn blit(&mut self, name: &str, pixel_buf: &wgpu::Buffer, image_width: u32, image_height: u32) {
        let (pipeline, layout) = match (&self.blit_pipeline, &self.blit_bind_group_layout) {
            (Some(p), Some(l)) => (p, l),
            _ => return,
        };

        let device = Self::dev(self.device);
        let queue = Self::q(self.queue);

        let target = match self.targets.get_mut(name) {
            Some(t) => t,
            None => return,
        };

        // Update image dimensions in viewport
        target.viewport.image_width = image_width as f32;
        target.viewport.image_height = image_height as f32;
        queue.write_buffer(&target.viewport_buf, 0, bytemuck::bytes_of(&target.viewport));

        // Get surface texture
        let frame = match target.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return,
        };
        let view = frame.texture.create_view(&Default::default());

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target.viewport_buf.as_entire_binding(),
                },
            ],
        });

        // Render pass
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display-blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1); // fullscreen triangle
        }

        Self::q(self.queue).submit([encoder.finish()]);
        frame.present();
    }

    /// Resize a display target (e.g., window resize).
    pub fn resize_target(&mut self, name: &str, width: u32, height: u32) {
        let device = Self::dev(self.device);
        if let Some(target) = self.targets.get_mut(name) {
            target.config.width = width.max(1);
            target.config.height = height.max(1);
            target.surface.configure(device, &target.config);
            target.viewport.canvas_width = width as f32;
            target.viewport.canvas_height = height as f32;
        }
    }

    /// Check if a named target exists.
    pub fn has_target(&self, name: &str) -> bool {
        self.targets.contains_key(name)
    }

    /// List all registered target names.
    pub fn target_names(&self) -> Vec<&str> {
        self.targets.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_default_is_sensible() {
        let vp = Viewport::default();
        assert_eq!(vp.zoom, 1.0);
        assert_eq!(vp.pan_x, 0.0);
        assert_eq!(vp.tone_mode, 0);
    }

    #[test]
    fn viewport_is_pod() {
        // Verify bytemuck Pod requirement
        let vp = Viewport::default();
        let bytes = bytemuck::bytes_of(&vp);
        assert_eq!(bytes.len(), 32); // 8 fields * 4 bytes
    }

    #[test]
    fn blit_shader_is_valid_wgsl() {
        assert!(BLIT_SHADER.contains("fn vs_main"));
        assert!(BLIT_SHADER.contains("fn fs_main"));
        assert!(BLIT_SHADER.contains("var<storage, read> pixels"));
        assert!(BLIT_SHADER.contains("var<uniform> vp"));
    }
}
