# Stroke Input Engine — Design Document

## 1. Per-Point Input Data Model

Professional stylus devices (Wacom, Apple Pencil, Samsung S Pen) report per-sample data at 120-240 Hz. The input model must capture all available data for maximum creative control.

### Sample Point Fields

| Field | Type | Unit | Range | Source |
|-------|------|------|-------|--------|
| `x` | f32 | pixels | 0..canvas_width | Pointer position |
| `y` | f32 | pixels | 0..canvas_height | Pointer position |
| `pressure` | f32 | normalized | 0.0..1.0 | Pen pressure sensor |
| `tilt_x` | f32 | radians | -PI/2..PI/2 | Pen tilt (azimuth component) |
| `tilt_y` | f32 | radians | -PI/2..PI/2 | Pen tilt (altitude component) |
| `rotation` | f32 | radians | 0..2*PI | Barrel rotation (Wacom Art Pen) |
| `velocity` | f32 | px/ms | 0..inf | Derived from position delta / time delta |
| `timestamp` | f64 | milliseconds | monotonic | Event timestamp (for velocity calc + replay) |

### Derived Fields (computed by brush engine, not input)

| Field | Derivation |
|-------|-----------|
| `direction` | atan2(dy, dx) between consecutive points |
| `curvature` | Second derivative of path (for smoothing) |
| `distance` | Cumulative arc length from stroke start |

### Data Layout

Points are stored as a flat `Vec<f32>` with a fixed stride per point. The stride is the number of fields * sizeof(f32). This makes the data GPU-uploadable without conversion.

```
[x0, y0, pressure0, tilt_x0, tilt_y0, rotation0, velocity0, timestamp0,
 x1, y1, pressure1, tilt_x1, tilt_y1, rotation1, velocity1, timestamp1,
 ...]
```

**Stride: 8 floats per point** (timestamp as f32 relative to stroke start for GPU, f64 absolute stored separately for replay).

For GPU upload, timestamp is stored as `f32` relative offset from stroke start. The absolute `f64` timestamps are kept host-side only for replay fidelity.

---

## 2. Brush Engine Architecture Survey

### 2.1 Krita Brush Engine

Krita has the most well-documented open-source brush engine. Key architecture:

**Stamp-based (dab) model:**
- A brush is a function that produces a "dab" (stamp image) given input parameters
- Dabs are placed along the stroke path at a configurable **spacing** interval (% of brush diameter)
- Each dab's properties (size, opacity, rotation, color) are modulated by **dynamics curves**
- Dynamics curves map input (pressure, velocity, tilt, distance, time, random) to output (size, opacity, rotation, scatter, flow)

**Brush tip types:**
- **Pixel mask** — loaded from PNG, rotated/scaled per dab
- **Parametric** — generated circle/ellipse with softness gradient
- **Predefined** — library of tip shapes (hair, chalk, watercolor)

**Key parameters:**
- `spacing` — distance between dabs as % of diameter (5-200%, typically 10-25%)
- `scatter` — random perpendicular displacement per dab
- `flow` — opacity per dab (vs `opacity` which is per-stroke)
- `hardness` — edge falloff (0=soft airbrush, 1=hard circle)
- `texture` — pattern overlaid on the dab (paper grain, canvas)
- `color dynamics` — hue/saturation/value shift per dab

**Painting modes:**
- Normal, Multiply, Screen, Overlay, etc. (standard blend modes)
- "Build up" vs "wash" mode — build-up accumulates on redraw, wash caps at brush opacity

**Performance:**
- Krita composites each dab onto a temporary "paint device" (layer buffer)
- Uses tiled memory (64x64 tiles), only allocating tiles that receive paint
- Multi-threaded per-tile compositing

### 2.2 Procreate

Procreate is the gold standard for mobile brush engines:

**Metal-accelerated stamp rendering:**
- Each dab is rendered as a textured quad via Metal instanced draw calls
- Brush texture is bound as a GPU texture; per-dab transform is a uniform
- Stroke is rendered to a dedicated "wet layer" then composited onto the canvas layer on stroke end

**Key innovations:**
- **Prediction** — renders predicted future points (Apple Pencil prediction API) for low-latency feel
- **Streamlining** — adjustable smoothing of input path (0-100%, Catmull-Rom or similar spline)
- **Wet paint mixing** — simulates paint mixing between current color and canvas color based on "wetness" parameter
- **Dual brush** — combines two brush tips with masking

**Brush tip:** stored as grayscale textures (typically 512x512 or 1024x1024), sampled per dab.

### 2.3 Photoshop

Photoshop's brush engine is the industry reference:

**Architecture:**
- Stamp-based with "transfer" (flow/opacity dynamics) and "shape dynamics" (size/angle/roundness)
- Scattering system: count, both axes, count jitter
- Texture integration: pattern overlay with blend mode (multiply, subtract, etc.)
- Color dynamics: foreground/background jitter, hue/saturation/brightness jitter
- Smoothing: 0-100% with "pulled string" mode (cursor trails behind; line is drawn from string, not cursor)

**GPU acceleration (CC 2018+):**
- Brush stamps rendered via GPU compute shaders on macOS (Metal) and Windows (D3D12/Vulkan)
- Texture sampling, transform, and compositing all GPU-side
- CPU fallback for exotic blend modes

---

## 3. GPU Rendering Architecture

### 3.1 Recommended Approach: GPU Stamp Instancing

The most practical GPU approach is **instanced stamp rendering** — one textured quad per dab, rendered in a single draw call.

**Pipeline:**

```
Input Points → Spacing Interpolation → Dab Generation → GPU Stamp Dispatch → Accumulation Buffer → Layer Composite
```

**Step 1: Spacing interpolation (CPU)**
Given raw input points, compute evenly-spaced dab positions along the stroke path. This runs on CPU because it's sequential and low-volume (typically <1000 dabs per stroke segment).

```
spacing_px = brush_diameter * spacing_percent
for each consecutive point pair:
    arc_length = distance(p0, p1)
    n_dabs = floor(arc_length / spacing_px)
    for each dab:
        t = (i * spacing_px - leftover) / arc_length
        pos = lerp(p0, p1, t)
        pressure = lerp(p0.pressure, p1.pressure, t)
        // ... interpolate all fields
```

**Step 2: Dab generation (CPU or GPU compute)**
For each dab position, compute the dab parameters:
- Size = base_size * pressure_curve(pressure) * size_dynamics(velocity, tilt, ...)
- Angle = base_angle + direction + rotation_dynamics(...)
- Opacity = flow * opacity_curve(pressure) * opacity_dynamics(...)
- Scatter offset = random perpendicular displacement

Output: array of `DabInstance` structs (position, size, angle, opacity).

**Step 3: GPU stamp dispatch**
Single compute shader pass that renders all dabs into an accumulation buffer:

```wgsl
struct DabInstance {
  pos: vec2<f32>,       // center position
  size: f32,            // diameter in pixels  
  angle: f32,           // rotation in radians
  opacity: f32,         // per-dab opacity
  color: vec3<f32>,     // dab color (after color dynamics)
}

// For each pixel in the accumulation buffer:
// 1. Transform to dab-local coordinates (rotate, scale)
// 2. Sample brush texture at transformed coords
// 3. Multiply by dab opacity and color
// 4. Composite onto accumulation buffer (max or additive)
```

**Two shader strategies:**

1. **Per-pixel iteration** (simpler, good for < 200 dabs):
   - Compute shader iterates over all dabs per output pixel
   - Each pixel checks which dabs overlap and accumulates
   - O(pixels * dabs) — acceptable for interactive stroke segments

2. **Instanced stamp blit** (better for many dabs):
   - Each dab dispatches a workgroup covering its bounding box
   - Workgroup reads brush texture, composites into accumulation buffer via atomics
   - O(sum of dab areas) — better scaling

**Recommendation: Start with per-pixel iteration** for simplicity. The V2 pipeline's compute shader model maps directly to this. Optimize to instanced blit if needed.

### 3.2 Accumulation Buffer

The accumulation buffer is a per-stroke temporary f32 RGBA buffer at canvas resolution. It holds the composited result of all dabs for the current stroke.

**Why a separate buffer:**
- Enables "wash" mode (cap opacity per-stroke at brush opacity)
- Enables stroke-level undo (discard buffer = undo)
- Enables incremental rendering (only composite new dabs, preserve old ones)

**Composite to layer:**
On stroke end (pointer up), the accumulation buffer is composited onto the target layer using the selected blend mode at the stroke-level opacity.

### 3.3 Incremental Rendering

For interactive performance, only render new stroke segments:

1. Track `last_rendered_dab_index` per stroke
2. On new input points: compute new dabs from `last_rendered_dab_index` onwards
3. Render only new dabs into the accumulation buffer (additive composite)
4. Composite the full accumulation buffer onto the display layer

This avoids re-rendering the entire stroke on every input event. For a typical 1000-point stroke at 240Hz, this means rendering 1-5 new dabs per frame instead of 1000.

---

## 4. Undo Model

### Stroke-Atomic Undo

Each stroke is a single undo unit. The undo stack stores:

```
UndoEntry {
    layer_id: u32,
    pre_stroke_snapshot: TileSet,  // only tiles modified by this stroke
    stroke_data: StrokeData,       // for replay if needed
}
```

**Tile-based snapshots:** Only capture tiles that the stroke's bounding box overlaps. For a typical 100px-wide stroke across a 4000x3000 canvas, this is ~50-100 tiles (64x64 each) = 800KB-1.6MB per undo entry (f32 RGBA), not the full canvas.

**Alternative: command-based undo** — store the stroke parameters and re-render from scratch on undo. Slower but uses less memory. Hybrid approach is typical: store tile snapshots up to a memory budget, fall back to re-render for old entries.

---

## 5. WIT Type Design

### Stroke Input Resource

```wit
/// A stroke being drawn interactively.
resource stroke-input {
    /// Create a new stroke with brush parameters.
    constructor(brush: brush-params);
    
    /// Add a sample point to the stroke.
    add-point: func(point: stroke-point);
    
    /// Signal end of stroke (pointer up).
    end-stroke: func();
    
    /// Get the number of points in this stroke.
    point-count: func() -> u32;
    
    /// Get serialized point data (f32 array, 8 floats per point).
    point-data: func() -> list<f32>;
}

/// Per-point input sample from a stylus/pointer device.
record stroke-point {
    x: f32,
    y: f32,
    pressure: f32,
    tilt-x: f32,
    tilt-y: f32,
    rotation: f32,
    velocity: f32,
    timestamp: f32,  // relative to stroke start (ms)
}

/// Brush configuration parameters.
record brush-params {
    /// Brush tip texture (grayscale, width x height).
    tip-texture: option<list<f32>>,
    tip-width: u32,
    tip-height: u32,
    
    /// Base diameter in pixels.
    diameter: f32,
    /// Spacing between dabs (% of diameter, e.g., 0.15 = 15%).
    spacing: f32,
    /// Edge hardness (0.0 = soft airbrush, 1.0 = hard circle).
    hardness: f32,
    /// Per-dab flow (opacity per dab, 0..1).
    flow: f32,
    /// Per-stroke opacity cap (0..1).
    opacity: f32,
    /// Base rotation angle (radians).
    angle: f32,
    /// Roundness (1.0 = circle, 0.1 = flat ellipse).
    roundness: f32,
    
    /// Dynamics: pressure curve for size (control points).
    pressure-size-curve: option<list<f32>>,
    /// Dynamics: pressure curve for opacity (control points).
    pressure-opacity-curve: option<list<f32>>,
    /// Dynamics: velocity curve for size.
    velocity-size-curve: option<list<f32>>,
    
    /// Scatter amount (0 = no scatter).
    scatter: f32,
    /// Scatter axis (true = both axes, false = perpendicular only).
    scatter-both-axes: bool,
    
    /// Color dynamics: hue jitter range (degrees).
    hue-jitter: f32,
    /// Color dynamics: saturation jitter range (0..1).
    saturation-jitter: f32,
    /// Color dynamics: brightness jitter range (0..1).
    brightness-jitter: f32,
    
    /// Smoothing amount (0..1, 0 = no smoothing).
    smoothing: f32,
}
```

### Pipeline Integration

The stroke input integrates as a filter parameter:

```wit
/// Apply a brush stroke to a layer.
/// The stroke is rendered to an internal accumulation buffer,
/// then composited onto the source layer.
apply-stroke: func(
    source: node-id,
    stroke: borrow<stroke-input>,
    color: tuple<f32, f32, f32>,
    blend-mode: string,
) -> result<node-id, rasmcore-error>;
```

### SDK API (TypeScript)

```typescript
const stroke = new StrokeInput({
  diameter: 20,
  spacing: 0.15,
  hardness: 0.8,
  flow: 0.9,
  opacity: 1.0,
  pressureSizeCurve: [0, 0, 0.5, 0.8, 1, 1],  // control points
});

// Called on pointermove events:
stroke.addPoint({
  x: event.clientX,
  y: event.clientY,
  pressure: event.pressure,
  tiltX: event.tiltX * Math.PI / 180,
  tiltY: event.tiltY * Math.PI / 180,
  rotation: event.twist * Math.PI / 180,
  velocity: computeVelocity(event),
  timestamp: event.timeStamp - strokeStartTime,
});

// Called on pointerup:
stroke.endStroke();

// Render:
const result = pipeline.open(canvas)
  .applyStroke(stroke, [1, 0, 0], 'normal')
  .render();
```

---

## 6. Implementation Tracks (Recommended)

If this design is approved, implementation would split into:

1. **stroke-input-wit** — Add WIT types, WASM adapter, SDK bindings
2. **brush-engine-cpu** — CPU stamp-based brush engine (spacing, dynamics, dab generation)
3. **brush-engine-gpu** — WGSL compute shader for stamp rendering + accumulation buffer
4. **brush-presets** — Standard brush preset library (pencil, pen, airbrush, watercolor, chalk)
5. **stroke-undo** — Tile-based undo stack with stroke-atomic entries

---

## 7. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Rendering model | Stamp-based (dab) | Industry standard (Krita, PS, Procreate). SDF accumulation is better for vector but worse for raster texture brushes. |
| GPU strategy | Per-pixel compute shader | Maps to V2 compute model. Start simple, optimize to instanced blit if needed. |
| Accumulation buffer | f32 RGBA, canvas-sized | Required for wash mode, stroke undo, incremental rendering. Memory cost is bounded by canvas size. |
| Undo model | Tile-based snapshot | Fast undo (swap tiles), bounded memory, no full-canvas copy. |
| Smoothing | Catmull-Rom spline | Well-understood, adjustable, used by Procreate. |
| Dynamics curves | Piecewise-linear control points | Same format as existing curves in grading filters. GPU-friendly (texture lookup). |
| Input model | 8 floats per point | Covers all major stylus capabilities. Fixed stride is GPU-friendly. |
| Brush tips | Grayscale f32 texture | Resolution-independent via bilinear sampling. Matches V2 f32 pipeline. |
