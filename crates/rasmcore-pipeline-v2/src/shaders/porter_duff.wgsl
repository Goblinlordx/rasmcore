// Porter-Duff "over" compositor GPU shader.
//
// Dual-input: load_pixel_a() = foreground, load_pixel_b() = background.
// Uses straight alpha — premultiplies internally for math, un-premultiplies output.
//
// Reference: Porter & Duff, "Compositing Digital Images" (1984), SIGGRAPH.

struct Params {
    width: u32, height: u32, opacity: f32, _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.x + gid.y * params.width;

    let fg = load_pixel_a(idx);
    let bg = load_pixel_b(idx);

    let fg_a = fg.a * params.opacity;
    let bg_a = bg.a;
    let inv_fg_a = 1.0 - fg_a;

    let out_a = fg_a + bg_a * inv_fg_a;

    var out_rgb: vec3<f32>;
    if (out_a > 0.0000001) {
        let inv_out_a = 1.0 / out_a;
        out_rgb = (fg.rgb * fg_a + bg.rgb * bg_a * inv_fg_a) * inv_out_a;
    } else {
        out_rgb = vec3(0.0);
    }

    store_pixel(idx, vec4(out_rgb, out_a));
}
