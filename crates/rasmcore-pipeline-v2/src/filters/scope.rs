//! Video scope render filters — analysis visualizations on black background.
//!
//! Each scope reads input f32 RGBA pixels and produces a fixed-size visualization
//! image (default 512x512). Output dimensions are independent of input dimensions.
//!
//! Scopes:
//! - `scope_histogram` — per-channel value distribution bars
//! - `scope_waveform` — per-column brightness distribution
//! - `scope_parade` — R/G/B side-by-side waveforms
//! - `scope_vectorscope` — chrominance polar plot (dots on black)

use crate::color_space::ColorSpace;
use crate::node::{Node, NodeInfo, PipelineError, Upstream};
use crate::rect::Rect;
use crate::registry::{
    FilterFactoryRegistration, OperationCapabilities, OperationRegistration,
    OperationKind, ParamDescriptor, ParamMap, ParamType,
};

// ─── ScopeNode wrapper ─────────────────────────────────────────────────────

/// Node wrapper for scope filters that changes output dimensions.
///
/// Unlike FilterNode (which preserves input dimensions), ScopeNode reports
/// a fixed `scope_size x scope_size` output. The filter receives the full
/// input and produces the scope image.
struct ScopeNode {
    upstream: u32,
    input_info: NodeInfo,
    scope_size: u32,
    compute_fn: fn(&[f32], u32, u32, u32, bool) -> Vec<f32>,
    log_scale: bool,
}

impl Node for ScopeNode {
    fn info(&self) -> NodeInfo {
        NodeInfo {
            width: self.scope_size,
            height: self.scope_size,
            color_space: ColorSpace::Srgb, // scope output is display-referred
        }
    }

    fn compute(
        &self,
        _request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        // Request full input (scopes analyze the entire image)
        let full = Rect::new(0, 0, self.input_info.width, self.input_info.height);
        let input = upstream.request(self.upstream, full)?;
        let output = (self.compute_fn)(
            &input,
            self.input_info.width,
            self.input_info.height,
            self.scope_size,
            self.log_scale,
        );
        Ok(output)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn input_rect(&self, _output: Rect, _bounds_w: u32, _bounds_h: u32) -> crate::node::InputRectEstimate {
        // Scopes need the full input image
        crate::node::InputRectEstimate::FullImage
    }
}

// ─── Rendering helpers ─────────────────────────────────────────────────────

/// Set a pixel in the scope buffer (additive blend for dot accumulation).
#[inline]
fn plot_dot(buf: &mut [f32], size: u32, x: i32, y: i32, r: f32, g: f32, b: f32, intensity: f32) {
    if x < 0 || y < 0 || x >= size as i32 || y >= size as i32 {
        return;
    }
    let idx = (y as usize * size as usize + x as usize) * 4;
    buf[idx] += r * intensity;
    buf[idx + 1] += g * intensity;
    buf[idx + 2] += b * intensity;
    // Alpha stays at 1.0 (set during init)
}

/// Draw a vertical bar from bottom (y=size-1) to given height.
fn fill_bar(buf: &mut [f32], size: u32, x: u32, height: u32, r: f32, g: f32, b: f32, alpha: f32) {
    let h = height.min(size);
    for dy in 0..h {
        let y = size - 1 - dy;
        let idx = (y as usize * size as usize + x as usize) * 4;
        // Additive blend for overlapping channels
        buf[idx] += r * alpha;
        buf[idx + 1] += g * alpha;
        buf[idx + 2] += b * alpha;
    }
}

/// Draw a line between two points (Bresenham).
fn draw_line(buf: &mut [f32], size: u32, x0: i32, y0: i32, x1: i32, y1: i32, r: f32, g: f32, b: f32, alpha: f32) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        plot_dot(buf, size, x0, y0, r, g, b, alpha);
        if x0 == x1 && y0 == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x0 += sx; }
        if e2 <= dx { err += dx; y0 += sy; }
    }
}

/// Create a black scope buffer with alpha = 1.0.
fn new_scope_buf(size: u32) -> Vec<f32> {
    let n = (size * size) as usize;
    let mut buf = vec![0.0f32; n * 4];
    // Set alpha to 1.0
    for i in 0..n {
        buf[i * 4 + 3] = 1.0;
    }
    buf
}

/// Clamp scope buffer values to [0, 1].
fn clamp_buf(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
}

// ─── RGB → HSL (local, for vectorscope) ────────────────────────────────────

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    if (max - min).abs() < 1e-7 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
    let h = if (max - r).abs() < 1e-7 {
        let mut h = (g - b) / d;
        if g < b { h += 6.0; }
        h * 60.0
    } else if (max - g).abs() < 1e-7 {
        ((b - r) / d + 2.0) * 60.0
    } else {
        ((r - g) / d + 4.0) * 60.0
    };
    (h, s, l)
}

// ─── Scope implementations ─────────────────────────────────────────────────

/// Histogram scope — 256-bin per-channel distribution.
fn compute_histogram(input: &[f32], _w: u32, _h: u32, size: u32, log_scale: bool) -> Vec<f32> {
    let mut bins_r = [0u32; 256];
    let mut bins_g = [0u32; 256];
    let mut bins_b = [0u32; 256];
    let mut bins_l = [0u32; 256];

    for pixel in input.chunks_exact(4) {
        let r = (pixel[0].clamp(0.0, 1.0) * 255.0) as usize;
        let g = (pixel[1].clamp(0.0, 1.0) * 255.0) as usize;
        let b = (pixel[2].clamp(0.0, 1.0) * 255.0) as usize;
        let luma = (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]).clamp(0.0, 1.0);
        let l = (luma * 255.0) as usize;
        bins_r[r.min(255)] += 1;
        bins_g[g.min(255)] += 1;
        bins_b[b.min(255)] += 1;
        bins_l[l.min(255)] += 1;
    }

    // Find max for normalization
    let max_count = bins_r.iter().chain(bins_g.iter()).chain(bins_b.iter())
        .copied().max().unwrap_or(1).max(1);

    let mut buf = new_scope_buf(size);

    for i in 0..256u32 {
        let x = (i as f32 / 255.0 * (size - 1) as f32) as u32;

        let normalize = |count: u32| -> u32 {
            let ratio = if log_scale {
                (count as f32 + 1.0).ln() / (max_count as f32 + 1.0).ln()
            } else {
                count as f32 / max_count as f32
            };
            (ratio * (size - 1) as f32) as u32
        };

        // Draw channels with partial transparency for overlap visibility
        fill_bar(&mut buf, size, x, normalize(bins_r[i as usize]), 0.9, 0.1, 0.1, 0.5);
        fill_bar(&mut buf, size, x, normalize(bins_g[i as usize]), 0.1, 0.9, 0.1, 0.5);
        fill_bar(&mut buf, size, x, normalize(bins_b[i as usize]), 0.1, 0.1, 0.9, 0.5);
        // Luma outline (draw on top, thin)
        let lh = normalize(bins_l[i as usize]);
        if lh > 0 {
            let y = (size - 1 - lh) as i32;
            plot_dot(&mut buf, size, x as i32, y, 0.7, 0.7, 0.7, 0.6);
        }
    }

    clamp_buf(&mut buf);
    buf
}

/// Waveform scope — per-column brightness distribution.
fn compute_waveform(input: &[f32], w: u32, h: u32, size: u32, _log_scale: bool) -> Vec<f32> {
    let mut buf = new_scope_buf(size);
    let intensity = (4.0 / h as f32).max(0.02).min(0.5); // brighter dots for fewer rows

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize * 4;
            let r = input[idx];
            let g = input[idx + 1];
            let b = input[idx + 2];
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            let sx = (x as f32 / w as f32 * (size - 1) as f32) as i32;
            let sy = ((1.0 - luma.clamp(0.0, 1.0)) * (size - 1) as f32) as i32;

            plot_dot(&mut buf, size, sx, sy, 0.2, 0.8, 0.2, intensity);
        }
    }

    clamp_buf(&mut buf);
    buf
}

/// Parade scope — R/G/B side-by-side waveforms.
fn compute_parade(input: &[f32], w: u32, h: u32, size: u32, _log_scale: bool) -> Vec<f32> {
    let mut buf = new_scope_buf(size);
    let third = size / 3;
    let intensity = (4.0 / h as f32).max(0.02).min(0.5);

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize * 4;
            let r = input[idx].clamp(0.0, 1.0);
            let g = input[idx + 1].clamp(0.0, 1.0);
            let b = input[idx + 2].clamp(0.0, 1.0);

            let sx = (x as f32 / w as f32 * (third - 2) as f32) as i32;

            // Red channel (left third)
            let ry = ((1.0 - r) * (size - 1) as f32) as i32;
            plot_dot(&mut buf, size, sx, ry, 0.9, 0.15, 0.15, intensity);

            // Green channel (middle third)
            let gy = ((1.0 - g) * (size - 1) as f32) as i32;
            plot_dot(&mut buf, size, sx + third as i32, gy, 0.15, 0.9, 0.15, intensity);

            // Blue channel (right third)
            let by = ((1.0 - b) * (size - 1) as f32) as i32;
            plot_dot(&mut buf, size, sx + 2 * third as i32, by, 0.15, 0.15, 0.9, intensity);
        }
    }

    // Separator lines
    draw_line(&mut buf, size, third as i32, 0, third as i32, size as i32 - 1, 0.3, 0.3, 0.3, 1.0);
    draw_line(&mut buf, size, 2 * third as i32, 0, 2 * third as i32, size as i32 - 1, 0.3, 0.3, 0.3, 1.0);

    clamp_buf(&mut buf);
    buf
}

/// Vectorscope — chrominance polar plot.
fn compute_vectorscope(input: &[f32], _w: u32, _h: u32, size: u32, _log_scale: bool) -> Vec<f32> {
    let mut buf = new_scope_buf(size);
    let center = size as f32 / 2.0;
    let radius = center - 2.0;
    let pixel_count = input.len() / 4;
    let intensity = (8.0 / pixel_count as f32).max(0.005).min(0.3);

    // Draw graticule: circle outline and color targets
    let steps = (size as f32 * std::f32::consts::PI) as i32;
    for i in 0..steps {
        let angle = i as f32 / steps as f32 * std::f32::consts::TAU;
        let x = (center + radius * angle.cos()) as i32;
        let y = (center - radius * angle.sin()) as i32;
        plot_dot(&mut buf, size, x, y, 0.2, 0.2, 0.2, 1.0);
    }
    // Crosshair
    draw_line(&mut buf, size, center as i32, 0, center as i32, size as i32 - 1, 0.15, 0.15, 0.15, 1.0);
    draw_line(&mut buf, size, 0, center as i32, size as i32 - 1, center as i32, 0.15, 0.15, 0.15, 1.0);

    // Skin tone line (~123° from positive I axis, roughly 33° from B-Y axis)
    let skin_angle = 123.0f32.to_radians();
    let sx = (center + radius * skin_angle.cos()) as i32;
    let sy = (center - radius * skin_angle.sin()) as i32;
    draw_line(&mut buf, size, center as i32, center as i32, sx, sy, 0.5, 0.4, 0.3, 0.6);

    // Color target markers (R, G, B, Cy, Mg, Yl at standard positions)
    let targets: [(f32, f32, f32, f32); 6] = [
        (0.0,   1.0, 0.0, 0.0),    // Red at 0°
        (120.0, 0.0, 1.0, 0.0),    // Green at 120°
        (240.0, 0.0, 0.0, 1.0),    // Blue at 240°
        (180.0, 0.0, 0.8, 0.8),    // Cyan at 180°
        (300.0, 0.8, 0.0, 0.8),    // Magenta at 300°
        (60.0,  0.8, 0.8, 0.0),    // Yellow at 60°
    ];
    for (angle_deg, tr, tg, tb) in targets {
        let a = angle_deg.to_radians();
        let tx = (center + radius * 0.75 * a.cos()) as i32;
        let ty = (center - radius * 0.75 * a.sin()) as i32;
        for dx in -2..=2i32 {
            for dy in -2..=2i32 {
                plot_dot(&mut buf, size, tx + dx, ty + dy, tr, tg, tb, 0.7);
            }
        }
    }

    // Plot each pixel's chrominance
    for pixel in input.chunks_exact(4) {
        let r = pixel[0].clamp(0.0, 1.0);
        let g = pixel[1].clamp(0.0, 1.0);
        let b = pixel[2].clamp(0.0, 1.0);
        let (h, s, _l) = rgb_to_hsl(r, g, b);

        if s < 0.001 { continue; } // skip achromatic

        let angle = h.to_radians();
        let dist = s * radius;
        let sx = (center + dist * angle.cos()) as i32;
        let sy = (center - dist * angle.sin()) as i32;

        // Dot color matches the pixel's color (dimmed)
        plot_dot(&mut buf, size, sx, sy, r * 0.6 + 0.2, g * 0.6 + 0.2, b * 0.6 + 0.2, intensity);
    }

    clamp_buf(&mut buf);
    buf
}

// ─── Registration ──────────────────────────────────────────────────────────

fn make_scope_node(
    upstream: u32,
    info: NodeInfo,
    params: &ParamMap,
    compute_fn: fn(&[f32], u32, u32, u32, bool) -> Vec<f32>,
) -> Box<dyn Node> {
    let scope_size = params.get_u32("scope_size");
    let scope_size = if scope_size == 0 { 512 } else { scope_size.clamp(64, 2048) };
    let log_scale = params.get_bool("log_scale");
    Box::new(ScopeNode { upstream, input_info: info, scope_size, compute_fn, log_scale })
}

// ─── Param descriptors ─────────────────────────────────────────────────────

static SCOPE_PARAMS: [ParamDescriptor; 2] = [
    ParamDescriptor {
        name: "scope_size",
        value_type: ParamType::U32,
        min: Some(64.0),
        max: Some(2048.0),
        step: Some(64.0),
        default: Some(512.0),
        hint: None,
        description: "Output scope image size (width and height in pixels)",
        constraints: &[],
    },
    ParamDescriptor {
        name: "log_scale",
        value_type: ParamType::Bool,
        min: None,
        max: None,
        step: None,
        default: Some(0.0),
        hint: None,
        description: "Use logarithmic scale for histogram bars",
        constraints: &[],
    },
];

// Histogram
static REG_HISTOGRAM: FilterFactoryRegistration = FilterFactoryRegistration {
    name: "scope_histogram",
    display_name: "Histogram",
    category: "analysis",
    params: &SCOPE_PARAMS,
    doc_path: "",
    cost: "O(n + s^2)",
    factory: |upstream, info, params| make_scope_node(upstream, info, params, compute_histogram),
};
inventory::submit!(&REG_HISTOGRAM);
static OPREG_HISTOGRAM: OperationRegistration = OperationRegistration {
    name: "scope_histogram", display_name: "Histogram", category: "analysis",
    kind: OperationKind::Filter, params: &SCOPE_PARAMS, doc_path: "",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: false },
    cost: "O(n + s^2)",
};
inventory::submit!(&OPREG_HISTOGRAM);

// Waveform
static REG_WAVEFORM: FilterFactoryRegistration = FilterFactoryRegistration {
    name: "scope_waveform",
    display_name: "Waveform",
    category: "analysis",
    params: &SCOPE_PARAMS,
    doc_path: "",
    cost: "O(n + s^2)",
    factory: |upstream, info, params| make_scope_node(upstream, info, params, compute_waveform),
};
inventory::submit!(&REG_WAVEFORM);
static OPREG_WAVEFORM: OperationRegistration = OperationRegistration {
    name: "scope_waveform", display_name: "Waveform", category: "analysis",
    kind: OperationKind::Filter, params: &SCOPE_PARAMS, doc_path: "",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: false },
    cost: "O(n + s^2)",
};
inventory::submit!(&OPREG_WAVEFORM);

// Parade
static REG_PARADE: FilterFactoryRegistration = FilterFactoryRegistration {
    name: "scope_parade",
    display_name: "Parade",
    category: "analysis",
    params: &SCOPE_PARAMS,
    doc_path: "",
    cost: "O(n + s^2)",
    factory: |upstream, info, params| make_scope_node(upstream, info, params, compute_parade),
};
inventory::submit!(&REG_PARADE);
static OPREG_PARADE: OperationRegistration = OperationRegistration {
    name: "scope_parade", display_name: "Parade", category: "analysis",
    kind: OperationKind::Filter, params: &SCOPE_PARAMS, doc_path: "",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: false },
    cost: "O(n + s^2)",
};
inventory::submit!(&OPREG_PARADE);

// Vectorscope
static REG_VECTORSCOPE: FilterFactoryRegistration = FilterFactoryRegistration {
    name: "scope_vectorscope",
    display_name: "Vectorscope",
    category: "analysis",
    params: &SCOPE_PARAMS,
    doc_path: "",
    cost: "O(n + s^2)",
    factory: |upstream, info, params| make_scope_node(upstream, info, params, compute_vectorscope),
};
inventory::submit!(&REG_VECTORSCOPE);
static OPREG_VECTORSCOPE: OperationRegistration = OperationRegistration {
    name: "scope_vectorscope", display_name: "Vectorscope", category: "analysis",
    kind: OperationKind::Filter, params: &SCOPE_PARAMS, doc_path: "",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: false },
    cost: "O(n + s^2)",
};
inventory::submit!(&OPREG_VECTORSCOPE);

/// Force-link all scope filter registrations.
///
/// Call this from the WASM crate to prevent the linker from dropping
/// the `inventory::submit!` statics. In native builds, `inventory`
/// uses linker sections that survive automatically; in WASM, statics
/// in sub-modules can be stripped as dead code.
pub fn ensure_linked() {
    // Reference each static to prevent dead-code elimination.
    // The actual values don't matter — just the references.
    let _ = &REG_HISTOGRAM;
    let _ = &OPREG_HISTOGRAM;
    let _ = &REG_WAVEFORM;
    let _ = &OPREG_WAVEFORM;
    let _ = &REG_PARADE;
    let _ = &OPREG_PARADE;
    let _ = &REG_VECTORSCOPE;
    let _ = &OPREG_VECTORSCOPE;
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_pixels(r: f32, g: f32, b: f32, count: usize) -> Vec<f32> {
        [r, g, b, 1.0].iter().copied().cycle().take(count * 4).collect()
    }

    /// Solid red image → histogram should have all weight in red bin 255.
    #[test]
    fn histogram_solid_red() {
        let pixels = solid_pixels(1.0, 0.0, 0.0, 100); // 10x10 red
        let scope = compute_histogram(&pixels, 10, 10, 256, false);
        assert_eq!(scope.len(), 256 * 256 * 4);
        // Red channel bar at x=255 should be max height, green/blue at x=255 should be zero
        // Just verify the output isn't all black
        let non_black: usize = scope.chunks_exact(4)
            .filter(|p| p[0] > 0.01 || p[1] > 0.01 || p[2] > 0.01)
            .count();
        assert!(non_black > 0, "histogram should have visible content");
    }

    /// Solid gray image → waveform should have a horizontal line at mid-height.
    #[test]
    fn waveform_solid_gray() {
        let pixels = solid_pixels(0.5, 0.5, 0.5, 100);
        let scope = compute_waveform(&pixels, 10, 10, 256, false);
        assert_eq!(scope.len(), 256 * 256 * 4);
        let non_black: usize = scope.chunks_exact(4)
            .filter(|p| p[0] > 0.01 || p[1] > 0.01 || p[2] > 0.01)
            .count();
        assert!(non_black > 0, "waveform should have visible content");
    }

    /// Parade of solid green → only middle third has content.
    #[test]
    fn parade_solid_green() {
        let pixels = solid_pixels(0.0, 1.0, 0.0, 100);
        let scope = compute_parade(&pixels, 10, 10, 256, false);
        assert_eq!(scope.len(), 256 * 256 * 4);
        let non_black: usize = scope.chunks_exact(4)
            .filter(|p| p[1] > 0.1) // green content
            .count();
        assert!(non_black > 0, "parade should have green content");
    }

    /// Vectorscope of saturated colors → dots away from center.
    #[test]
    fn vectorscope_saturated() {
        let mut pixels = Vec::with_capacity(400);
        for _ in 0..25 { pixels.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]); } // red
        for _ in 0..25 { pixels.extend_from_slice(&[0.0, 1.0, 0.0, 1.0]); } // green
        for _ in 0..25 { pixels.extend_from_slice(&[0.0, 0.0, 1.0, 1.0]); } // blue
        for _ in 0..25 { pixels.extend_from_slice(&[0.5, 0.5, 0.5, 1.0]); } // gray (no chroma)
        let scope = compute_vectorscope(&pixels, 10, 10, 256, false);
        assert_eq!(scope.len(), 256 * 256 * 4);
        // Should have dots away from center (saturated colors) plus graticule
        let non_black: usize = scope.chunks_exact(4)
            .filter(|p| p[0] > 0.01 || p[1] > 0.01 || p[2] > 0.01)
            .count();
        assert!(non_black > 50, "vectorscope should have visible dots and graticule");
    }

    /// Scope size parameter produces correct output dimensions.
    #[test]
    fn scope_size_parameter() {
        let pixels = solid_pixels(0.5, 0.5, 0.5, 100); // 10x10 image
        for size in [64, 128, 256, 512] {
            let scope = compute_histogram(&pixels, 10, 10, size, false);
            assert_eq!(scope.len(), (size * size * 4) as usize, "scope size {size}");
        }
    }

    /// Log scale doesn't crash and produces different output than linear.
    #[test]
    fn histogram_log_scale() {
        // Varied input with many different values — log and linear normalization differ
        let mut pixels = Vec::with_capacity(4000);
        for i in 0..1000 {
            let v = (i as f32) / 999.0;
            pixels.extend_from_slice(&[v, v * 0.5, v * 0.3, 1.0]);
        }
        let linear = compute_histogram(&pixels, 100, 10, 128, false);
        let log = compute_histogram(&pixels, 100, 10, 128, true);
        assert_eq!(linear.len(), log.len());
        assert_ne!(linear, log, "log scale should differ from linear");
    }

    /// Scopes are discoverable via registry.
    #[test]
    fn scopes_registered() {
        let names: Vec<&str> = crate::registered_operations()
            .into_iter()
            .filter(|op| op.category == "analysis")
            .map(|op| op.name)
            .collect();
        assert!(names.contains(&"scope_histogram"), "histogram not registered");
        assert!(names.contains(&"scope_waveform"), "waveform not registered");
        assert!(names.contains(&"scope_parade"), "parade not registered");
        assert!(names.contains(&"scope_vectorscope"), "vectorscope not registered");
    }

    /// Factory creates ScopeNode with correct output dimensions.
    #[test]
    fn factory_creates_scope_node() {
        let info = NodeInfo { width: 100, height: 100, color_space: ColorSpace::Linear };
        let mut params = ParamMap::new();
        params.ints.insert("scope_size".into(), 256);
        let node = crate::create_filter_node("scope_histogram", 0, info, &params).unwrap();
        let node_info = node.info();
        assert_eq!(node_info.width, 256, "scope node should report scope_size as width");
        assert_eq!(node_info.height, 256, "scope node should report scope_size as height");
    }
}
