# rasmcore demo

Interactive demo of rasmcore image processing running entirely in the browser via WebAssembly.

## Quick Start

```bash
# 1. Build WASM + generate browser SDK
./demo/build.sh

# 2. Serve locally (any static server works)
npx serve demo/

# 3. Open http://localhost:3000/pipeline.html
```

## Pages

| Page | Description |
|------|-------------|
| `pipeline.html` | Pipeline builder — visual chain editor with auto-discovered operations |
| `index.html` | Simple demo — single-image filter sliders |
| `smoke-test.html` | SDK smoke test — verifies WASM loads correctly |

## Pipeline Builder Features

- **Auto-discovered operations** from `param-manifest.json` (generated at build time from `#[param]` attributes)
- **Dynamic parameter controls** — sliders for numbers, dropdowns for enums, color pickers for `rc.color_rgb` hints
- **Drag-to-reorder** operation nodes in the chain
- **Edit mode** with debounced thumbnail preview (1-second delay)
- **Before/After toggle** to compare original vs processed
- **Multi-image layers** — load multiple images, composite with blend modes (multiply, screen, overlay, etc.) and x/y offset
- **Per-layer filter chains** — each layer can have its own processing chain
- **Web Worker** — all WASM processing runs off the main thread (zero UI jank)
- **Single-slot queue** — new requests replace pending, never accumulate
- **Export** with auto-discovered format options + quality slider
- **Get Code** button generates SDK code (single-image or multi-image with `loadLayer`/`composite`)
- **Per-operation timing** displayed on each node card

## Build Chain

```
cargo component build -p rasmcore-image --release
    -> target/wasm32-wasip1/release/rasmcore_image.wasm

npx jco transpile rasmcore_image.wasm -o demo/sdk/
    -> demo/sdk/rasmcore-image.js       (ESM wrapper)
    -> demo/sdk/rasmcore-image.d.ts     (TypeScript types)
    -> demo/sdk/rasmcore-image.core.wasm (core module)
    -> demo/sdk/param-manifest.json     (auto-generated param metadata)
```

## Fluent SDK

```typescript
import { rcimage } from '@rasmcore/sdk';

// Single image
const result = rcimage.load(bytes, mod)
  .blur(3.0)
  .resize(800, 600, 'lanczos3')
  .toJpeg({ quality: 85 });

// Multi-image composite
const bg = rcimage.load(bgBytes, mod);
const fg = bg.loadLayer(fgBytes).blur(2.0);
bg.composite(fg, { x: 10, y: 10, blend: 'multiply' })
  .toJpeg({ quality: 85 });

// Metadata (read breaks chain, write is chainable)
const meta = await rcimage.load(bytes, mod).metadata();
meta.dump();                // { exif: { Artist: '...' }, xmp: {...} }
meta.read('exif.Artist');   // 'John Doe'

rcimage.load(bytes, mod)
  .keepMetadata()
  .stripMetadata('exif.GPSLatitude')
  .setMetadata('exif.Artist', 'rasmcore')
  .toJpeg({ quality: 85 });
```

### SDK Generator

```bash
# Generate JS/TS SDK (camelCase — default)
node scripts/generate-fluent-sdk.cjs --naming camel

# Generate Python-style SDK (snake_case)
node scripts/generate-fluent-sdk.cjs --naming snake
```

## Param Hints

Filters declare UI rendering hints via `#[param(hint = "rc.color_rgb")]`.
Standard `rc.` hints are rendered by the demo UI; third parties can use
any prefix (`myapp.gradient_stops`) for their own custom UIs.

| Hint | Type | UI Control |
|------|------|------------|
| `rc.color_rgb` | `[u8; 3]` | Color picker |
| `rc.color_rgba` | `[u8; 4]` | Color picker + alpha |
| `rc.angle_deg` | `f32` | Slider 0-360 |
| `rc.percentage` | `f32` | Slider 0-100% |
| `rc.kernel` | `&[f32]` | Matrix editor |
| `rc.curve` | `&[(f32,f32)]` | Curve editor |
| `rc.point` | `(u32,u32)` | Coordinate picker |
| `rc.rect` | `(u32,u32,u32,u32)` | Rectangle picker |
| `rc.text` | `String` | Text input |

## Architecture

- **WASM Component**: Rust -> wasm32-wasip1 via `cargo component`
- **JS Bindings**: jco transpile generates ESM module + core WASM
- **Web Worker**: all pipeline processing runs off-thread via `pipeline-worker.js`
- **Single-slot queue**: latest request replaces pending, never accumulates
- **No framework**: vanilla JS/TS, works with any bundler or `<script type="module">`
- **Lazy pipeline**: operations build a DAG; pixels computed only on write()
