# rasmcore demo

Interactive demo of rasmcore image processing running entirely in the browser via WebAssembly.

## Quick Start

```bash
# 1. Build WASM + generate browser SDK
./demo/build.sh

# 2. Serve locally (any static server works)
npx serve demo/

# 3. Open http://localhost:3000/smoke-test.html
```

## Build Chain

```
cargo component build -p rasmcore-image --release
    → target/wasm32-wasip1/release/rasmcore_image.wasm (11MB)

npx @bytecodealliance/jco transpile rasmcore_image.wasm -o demo/sdk/
    → demo/sdk/rasmcore-image.js       (789KB, ESM wrapper)
    → demo/sdk/rasmcore-image.d.ts     (TypeScript types)
    → demo/sdk/rasmcore-image.core.wasm (11MB, core module)
    → demo/sdk/interfaces/*.d.ts        (per-interface types)
```

## SDK API

```typescript
import { pipeline, decoder, encoder, filters, transform } from './sdk/rasmcore-image.js';

// Pipeline (lazy, chainable via node IDs)
const pipe = new pipeline.ImagePipeline();
const src = pipe.read(pngBytes);
const blurred = pipe.blur(src, 3.0);
const resized = pipe.resize(blurred, 800, 600, 'lanczos3');
const output = pipe.writeJpeg(resized, { quality: 85 }, undefined);

// Stateless (immediate execution)
const decoded = decoder.decode(jpegBytes);
const blurredPixels = filters.blur(decoded.pixels, decoded.info, 3.0);
const jpegBytes = encoder.encodeJpeg(blurredPixels, decoded.info, { quality: 85 }, undefined);
```

## Architecture

- **WASM Component**: Rust → wasm32-wasip1 via `cargo component`
- **JS Bindings**: jco transpile generates ESM module + core WASM
- **No framework**: vanilla JS/TS, works with any bundler or `<script type="module">`
- **Lazy pipeline**: operations build a DAG; pixels computed only on write()

## Future: Fluent SDK

```typescript
import { rcimage } from '@rasmcore/sdk';

const result = await rcimage.load(bytes)
  .blur(3.0)
  .resize(800, 600, 'lanczos3')
  .brightness(0.1)
  .toJpeg({ quality: 85 });
```

See the `demo-fluent-sdk` track for the chainable wrapper API.
