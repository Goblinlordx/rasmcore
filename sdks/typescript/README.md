# @rasmcore/image — TypeScript SDK

Image processing SDK powered by WebAssembly. Load any rasmcore-compatible
WASM module and apply filters, transforms, and encoders.

## Install

```bash
npm install @rasmcore/image
```

## Usage (Dynamic — works with any module)

```typescript
import { RcImage } from '@rasmcore/image';
import * as rasmcore from './rasmcore-image.js'; // jco-transpiled WASM

const img = RcImage.load(rasmcore, pngBytes);
const jpeg = img
  .apply('blur', { radius: 3.0 })
  .apply('brightness', { amount: 0.1 })
  .encode('jpeg', { quality: 85 });
```

No code generation needed. The runtime reads the module's embedded manifest
to discover available operations and validate parameters.

## Usage (Typed — optional, full autocomplete)

Generate typed wrappers from your specific WASM module:

```bash
npx rcimg-typegen ./rasmcore_image.wasm --output src/rasmcore-typed.ts
```

```typescript
import { RcImage } from './rasmcore-typed';
import * as rasmcore from './rasmcore-image.js';

const jpeg = RcImage.load(rasmcore, pngBytes)
  .blur(3.0)           // ← autocomplete + type checking
  .brightness(0.1)     // ← param ranges in JSDoc
  .toJpeg({ quality: 85 });
```

The typed layer extends the dynamic runtime — same WASM calls, just with
a typed surface. If the module changes, regenerate to pick up new operations.

## Introspection

```typescript
const img = RcImage.load(rasmcore, pngBytes);
console.log(img.availableOperations); // ['blur', 'brightness', 'swirl', ...]
console.log(img.operationMeta('blur')); // { params: [{ name: 'radius', min: 0, max: 100, ... }] }
console.log(img.manifestHash); // '0678a1ace1ec5dde'
```
