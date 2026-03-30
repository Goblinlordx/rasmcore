# Parameter Hints — `rc.*` Vocabulary

Parameter hints tell the UI how to render filter controls. They are metadata
only — they do not affect runtime behavior, WIT signatures, or adapter code.

## Hint Levels

### Type-level hint (`#[config_hint("...")]`)

Put on a reusable ConfigParams struct. Propagates to all fields when embedded.

```rust
#[derive(ConfigParams)]
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
#[derive(ConfigParams)]
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

| Hint | UI Control | Field types | Grouping |
|------|-----------|-------------|----------|
| `rc.color_rgb` | Color picker (no alpha) | u8 or f32 | Fields sharing prefix + hint grouped as one picker |
| `rc.color_rgba` | Color picker (with alpha) | u8 or f32 | Fields sharing prefix + hint grouped as one picker |
| `rc.angle_deg` | Angle dial (0-360) | f32 | Single field |
| `rc.percentage` | Percentage slider (0-100) | f32 | Single field |

### Grouping convention

For field-level hints on separate `_r/_g/_b` fields, the UI groups fields
that share the same name prefix AND the same hint into one control.

Example: `slope_r`, `slope_g`, `slope_b` all with `hint = "rc.color_rgb"`
→ grouped into one color picker labeled "slope".

## Adding a New Param Type

1. Add the struct to `param_types.rs` with `#[config_hint("rc.new_hint")]`
2. Add the hint to the demo UI mapping in `demo/src/pipeline.js`
3. Document the hint here
4. Use the type in filter ConfigParams structs
