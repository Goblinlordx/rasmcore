# V2 Pipeline Migration Guide

## Overview

The V2 pipeline is a ground-up rebuild replacing V1 with a clean, f32-native architecture. V1 patterns are deprecated — all new development uses V2.

## What Changed

| Aspect | V1 | V2 |
|---|---|---|
| **Pixel data** | `&[u8]` + PixelFormat (20 variants) | `&[f32]` — always RGBA, 4 channels |
| **Format dispatch** | 1,621 PixelFormat references | Zero |
| **GPU buffers** | U32Packed + F32Vec4 | Always F32Vec4 |
| **GPU priority** | CPU primary, GPU optional | GPU primary, CPU fallback |
| **Filter registration** | `#[register_filter]` + `#[derive(ConfigParams)]` | `#[derive(Filter)]` + `impl CpuFilter` |
| **Config access** | `config.field` | `self.field` |
| **Color pipeline** | Decorative ColorSpace field | ACES-aware, per-node tracking, auto-conversion |
| **Operation fusion** | u8 LUT composition | Expression tree IR + WGSL codegen |
| **Cache** | SpatialCache only | SpatialCache + LayerCache with Q16/Q8 quantization |

## V2 Filter Pattern

### Before (V1)

```rust
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct BrightnessParams {
    #[param(min = -1.0, max = 1.0, default = 0.0)]
    pub amount: f32,
}

#[rasmcore_macros::register_filter(name = "brightness", category = "adjustment")]
pub fn brightness(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &BrightnessParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    // format dispatch: is_f32? is_16bit? process_via_8bit?
    // ...
}
```

### After (V2)

```rust
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "brightness", category = "adjustment")]
pub struct Brightness {
    #[param(min = -1.0, max = 1.0, default = 0.0)]
    pub amount: f32,
}

impl CpuFilter for Brightness {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let pixels = upstream(request)?;
        // No format dispatch — process f32 directly
        // self.amount instead of config.amount
    }
}
```

### Key Differences

1. `#[derive(ConfigParams)]` + `#[register_filter]` → `#[derive(Filter)]` + `#[filter(...)]`
2. Struct fields ARE params (no separate ConfigParams struct)
3. `impl CpuFilter` replaces bare function
4. `self.field` replaces `config.field`
5. No format dispatch (`validate_format`, `is_16bit`, `process_via_8bit` removed)
6. Optional traits: `GpuFilter`, `PointOp`, `ColorOp`, `AnalyticOp`

## Migration Checklist

For each filter:

1. Change `#[derive(ConfigParams, Clone)]` to `#[derive(Filter, Clone)]`
2. Add `#[filter(name = "...", category = "...")]` with attrs from old `#[register_filter]`
3. Remove the `#[register_filter]` function
4. Add `impl CpuFilter for StructName` with the function body
5. Replace `config.field` with `self.field`
6. Replace `upstream: &mut UpstreamFn` with `upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_)`
7. Remove format dispatch calls (`validate_format`, `is_f32`, `process_via_8bit`, etc.)
8. If `InputRectProvider` was implemented, move to `CpuFilter::input_rect()`

## V1 Deprecation

The following V1 APIs are deprecated and will be removed:

- `#[register_filter]` macro (use `#[derive(Filter)]`)
- `#[derive(ConfigParams)]` (struct fields are params now)
- `PixelFormat` dispatch in filter logic
- `process_via_8bit` / `process_via_standard`
- `validate_format` / `is_16bit` / `is_float` in filter bodies
- `BufferFormat::U32Packed` GPU buffers
- `io_u32.wgsl` / `pixel_ops.wgsl` GPU fragments

## What's NOT Migrated

Generators, mappers, and compositors still use V1 registration macros:
- `#[register_generator]` — no input image, returns `Vec<u8>`
- `#[register_mapper]` — changes pixel format, returns `(Vec<u8>, ImageInfo)`
- `#[register_compositor]` — combines two images

These have different signatures that don't map to the `CpuFilter` trait.
V2 equivalents may be added in the future.
