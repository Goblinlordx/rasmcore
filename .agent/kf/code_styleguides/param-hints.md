# Parameter Hints — `rc.*` Vocabulary

Parameter hints tell the UI how to render filter controls. They are metadata
only — they do not affect runtime behavior, WIT signatures, or adapter code.

## Hint Levels

### Type-level hint (`#[config_hint("...")]`)

Put on a reusable param type struct. Propagates to all fields when embedded.

```rust
#[derive(rasmcore_macros::ConfigParams)]
#[config_hint("rc.color_rgba")]
pub struct ColorRgba {
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub r: u8,
    // ...
}
```

### Field-level hint (`#[param(hint = "...")]`)

Put on individual fields. Overrides the type-level hint when both are present.

```rust
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "split_toning", category = "grading")]
pub struct SplitToningParams {
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 40.0, hint = "rc.angle_deg")]
    pub highlight_hue: f32,
}
```

## Reusable Param Types

Defined in `crates/rasmcore-image/src/domain/param_types.rs`:

| Type | Hint | Fields | Use for |
|------|------|--------|---------|
| `ColorRgba` | `rc.color_rgba` | r, g, b, a: u8 | Drawing colors with alpha |
| `ColorRgb` | `rc.color_rgb` | r, g, b: u8 | Tint/blend target colors |

Embed these as fields in your ConfigParams struct — the hint propagates
automatically. No `#[param(hint = "...")]` needed.

## Hint Vocabulary

### Color & Grouping

| Hint | UI Control | When to use |
|------|-----------|-------------|
| `rc.color_rgb` | Color picker (no alpha) | RGB triplet fields grouped as one picker |
| `rc.color_rgba` | Color picker (with alpha) | RGBA fields grouped as one picker |

### Specialized Input Types

| Hint | UI Control | When to use | Reference |
|------|-----------|-------------|-----------|
| `rc.angle_deg` | Angle dial or slider (0-360) | Hue rotation, band angle, blade rotation | GIMP/PS hue slider |
| `rc.percentage` | Slider (0-100) with % label | Normalized strength/amount as user-facing % | PS opacity control |
| `rc.opacity` | Thin slider + text input (0.0-1.0) | Blend amount, tint strength, factor | PS layer opacity |
| `rc.signed_slider` | Bipolar slider centered at 0 | Params with -N to +N range (brightness, balance, offset) | PS brightness/contrast |
| `rc.log_slider` | Logarithmic slider | Blur radius, sigma — low values need precision | GIMP unsharp mask radius |
| `rc.pixels` | Number spinner (no slider) | Pixel coordinates, image dimensions (large ranges) | PS canvas size inputs |
| `rc.spinner` | Number input with up/down arrows | Any precise numeric param better as input than slider | GIMP filter param spinners |
| `rc.seed` | Number input only (no slider) | Random seeds — slider meaningless for large integer ranges | — |
| `rc.temperature_k` | Slider with blue-to-warm gradient | Color temperature in Kelvin (2000-12000K) | Lightroom WB slider |
| `rc.toggle` | Toggle switch | Boolean on/off params | PS checkbox options |
| `rc.text` | Text input field | String params (JSON arrays, paths) | — |
| `rc.enum` | Dropdown select | Mode/shape/method (when param encodes a choice) | PS blend mode dropdown |

### Canvas Interaction Hints

| Hint | UI Control | When to use | Reference |
|------|-----------|-------------|-----------|
| `rc.point` | Canvas point selector | Click-to-place coordinates (cx/cy pairs) | PS eyedropper tool |
| `rc.path` | Canvas path drawer | Brush stroke coordinates (freehand drawing) | PS brush tool |
| `rc.box_select` | Canvas rectangle drag | Crop/selection region (width/height pairs) | PS marquee tool |

### Decision Guide

```
Is it a boolean?           → rc.toggle
Is it a color?             → rc.color_rgb / rc.color_rgba (via param type)
Is it an angle in degrees? → rc.angle_deg
Is it a pixel coord/dim?   → rc.pixels (or rc.point for canvas interaction)
Is it a canvas coordinate? → rc.point (paired x/y params)
Is it a brush path?        → rc.path
Is it a selection region?  → rc.box_select (paired w/h params)
Is it a random seed?       → rc.seed
Is it a temperature?       → rc.temperature_k
Is it a string?            → rc.text
Is it 0.0-1.0 opacity?    → rc.opacity
Is it -N to +N centered?  → rc.signed_slider
Is it a radius/sigma?     → rc.log_slider (if impact is non-linear at low values)
Is the range small (< 100) and linear?  → no hint needed (default slider is fine)
Is the range large (> 1000)?            → rc.spinner or rc.pixels
Otherwise                  → rc.spinner for precision, or leave as default slider
```

### Grouping Convention

For field-level hints on separate `_r/_g/_b` fields, the UI groups fields
that share the same name prefix AND the same hint into one control.

Example: `slope_r`, `slope_g`, `slope_b` all with `hint = "rc.color_rgb"`
→ grouped into one color picker labeled "slope".

## Enum Options with Descriptions

For `rc.enum` params, use the `options` attribute to provide per-choice
descriptions. These are emitted in `param-manifest.json` as an `options`
array that the UI can render as tooltips, info panels, or rich dropdowns.

### Syntax

```rust
#[derive(ConfigParams)]
pub struct ColorizeParams {
    /// Colorize method
    #[param(
        default = "w3c",
        hint = "rc.enum",
        options = "w3c:Photoshop/W3C standard — SetLum/ClipColor|lab:CIELAB perceptual — parabolic weighting"
    )]
    pub method: String,
}
```

### Format

`options = "value1:description1|value2:description2|..."`

- `|` separates options (not comma, to avoid conflicts with description text)
- `:` separates the value from its description
- Values should be short identifiers; descriptions can be full sentences

### Manifest Output

```json
{
  "name": "method",
  "hint": "rc.enum",
  "options": [
    {"value": "w3c", "description": "Photoshop/W3C standard — SetLum/ClipColor"},
    {"value": "lab", "description": "CIELAB perceptual — parabolic weighting"}
  ]
}
```

The `options` key is only present when the attribute is set. Params without
`options` emit no `options` key (backwards compatible).

## Adding a New Hint Type

1. Document the hint in this file (vocabulary table + decision guide)
2. Add rendering logic to `demo/src/pipeline.js` (hint → control mapping)
3. Add CSS for the new control type to `demo/pipeline.html`
4. Use the hint in `#[param(hint = "...")]` on appropriate filter params
