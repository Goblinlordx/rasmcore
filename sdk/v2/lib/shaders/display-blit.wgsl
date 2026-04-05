// Fullscreen-triangle blit shader: storage buffer → canvas texture
//
// Reads f32 pixel data from the compute output storage buffer and writes
// to the canvas texture. Supports pan/zoom via viewport uniforms.

struct Viewport {
    // Canvas dimensions in pixels
    canvas_width: f32,
    canvas_height: f32,
    // Image dimensions in pixels
    image_width: f32,
    image_height: f32,
    // Pan offset in image pixels (center of viewport in image space)
    pan_x: f32,
    pan_y: f32,
    // Zoom factor (1.0 = fit, 2.0 = 200%, etc.)
    zoom: f32,
    // Tone mapping mode: 0 = standard (clamp), 1 = extended (pass-through)
    tone_mode: u32,
};

@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> vp: Viewport;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle — 3 vertices, no vertex buffer needed.
// Oversized triangle clipped to viewport; more efficient than a quad.
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
    // Map UV (canvas space) → image pixel coordinate
    let canvas_px = in.uv * vec2<f32>(vp.canvas_width, vp.canvas_height);

    // Canvas center
    let center = vec2<f32>(vp.canvas_width, vp.canvas_height) * 0.5;

    // Image pixel coordinate (accounting for pan + zoom)
    let img_center = vec2<f32>(vp.image_width, vp.image_height) * 0.5;
    let img_px = (canvas_px - center) / vp.zoom + img_center + vec2<f32>(vp.pan_x, vp.pan_y);

    // Bounds check
    let ix = i32(floor(img_px.x));
    let iy = i32(floor(img_px.y));
    let w = i32(vp.image_width);
    let h = i32(vp.image_height);

    if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Nearest-neighbor sampling from storage buffer
    let idx = iy * w + ix;
    var color = pixels[idx];

    // Tone mapping
    if (vp.tone_mode == 0u) {
        color = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
    }
    // Extended mode: pass through — canvas toneMapping handles display mapping

    // Premultiply alpha for canvas compositing
    color = vec4<f32>(color.rgb * color.a, color.a);

    return color;
}
