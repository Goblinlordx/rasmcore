# Shared Type Design & Interface Governance

## Research Date: 2026-03-27

---

## Buffer and Memory Sharing Between Components

### The Fundamental Constraint

Components have **isolated linear memories**. There is NO shared memory between components. All data passes through the Canonical ABI:

1. Caller serializes data from its memory
2. Canonical ABI copies data across the component boundary
3. Callee deserializes into its own memory

**This means:** Passing a `list<u8>` (our buffer type) copies the bytes. For a 100MB image, this is a 100MB copy at each component boundary crossing.

### Implications for razm/core

This is the **"Context Switch Tax"** mentioned in the original conversation. The solution is **coarse-grained APIs**:

| Bad (Fine-Grained) | Good (Coarse-Grained) |
|---------------------|----------------------|
| `get-pixel(x, y) -> pixel` | `process(image-buffer) -> image-buffer` |
| `push-byte(b: u8)` | `push-frame(frame: list<u8>)` |
| Millions of cross-boundary calls | Single call per operation |

### Stream-Based Alternative

For large data, use streams instead of buffers:
```wit
// Host provides input-stream (file, network, etc.)
// Component reads chunks internally — no copy overhead per chunk
process-stream: func(input: input-stream) -> output-stream;
```

Streams avoid the copy problem because the component reads directly from the host-provided stream handle. Data is copied in chunks (e.g., 64KB) rather than all at once.

### WASI 0.3 Improvement

With 0.3's native `stream<T>`:
```wit
// Typed stream — component reads frames as they arrive
process-frames: async func(input: stream<video-frame>) -> stream<video-frame>;
```

This is the ideal pattern for video/audio processing pipelines.

---

## Resource Types and Ownership

### When to Use Resources vs Records

| Use Case | Type | Why |
|----------|------|-----|
| Immutable data passed in/out | Record | No lifecycle needed |
| Configuration | Record | Passed once, no state |
| Stateful processing | Resource | Has lifecycle (create → use → drop) |
| Handles to host objects | Resource | Represents external state |
| Streams, file descriptors | Resource | WASI convention |

### Resource Ownership Rules

```wit
resource encoder {
    constructor(config: encoder-config);

    // Takes ownership of frame data (consumes it)
    push-frame: func(data: list<u8>) -> result<_, error>;

    // Borrows encoder state (doesn't consume it)
    get-stats: func() -> encoder-stats;

    // Static function — no instance needed
    supported-profiles: static func() -> list<string>;
}
```

- **Owned parameters**: Caller transfers ownership, cannot use after call
- **Borrowed parameters**: `borrow<T>`, temporary loan, caller retains ownership
- **Drop**: When last owned reference dropped, resource is destroyed

---

## Proposed Shared Type Architecture for razm/core

### Package Structure

```
razm:core@0.1.0       — Shared types and error definitions
razm:image@0.1.0      — Image processing interfaces
razm:video@0.1.0      — Video processing interfaces
razm:data@0.1.0       — Data processing interfaces
razm:codec@0.1.0      — Codec plugin interfaces
```

### Core Shared Types (razm:core)

```wit
package razm:core@0.1.0;

interface types {
    /// Raw byte buffer — the universal data container
    type buffer = list<u8>;

    /// Pixel format for image data
    enum pixel-format {
        rgb8,
        rgba8,
        bgr8,
        bgra8,
        gray8,
        gray16,
        yuv420p,
        yuv422p,
        yuv444p,
    }

    /// Color space
    enum color-space {
        srgb,
        linear-srgb,
        display-p3,
        bt709,
        bt2020,
    }

    /// Sample format for audio data
    enum sample-format {
        f32-le,
        s16-le,
        s32-le,
    }

    /// Image dimensions and format metadata
    record image-info {
        width: u32,
        height: u32,
        format: pixel-format,
        color-space: color-space,
    }

    /// Video frame metadata
    record frame-info {
        width: u32,
        height: u32,
        format: pixel-format,
        timestamp-us: s64,
        keyframe: bool,
    }

    /// Audio buffer metadata
    record audio-info {
        sample-rate: u32,
        channels: u16,
        format: sample-format,
        samples: u32,
    }
}

interface errors {
    /// Universal error type for razm operations
    variant razm-error {
        invalid-input(string),
        unsupported-format(string),
        codec-error(string),
        io-error(string),
        out-of-memory,
        not-implemented,
    }
}
```

### Minimal Cross-Domain Types

The shared types should be **minimal** — only what multiple domains actually need:

| Shared Type | Used By |
|-------------|---------|
| `buffer` (`list<u8>`) | All domains |
| `pixel-format` | Image, Video |
| `color-space` | Image, Video |
| `sample-format` | Video (audio track), Audio |
| `image-info` | Image, Video (frame extraction) |
| `frame-info` | Video |
| `audio-info` | Video (audio track), Audio |
| `razm-error` | All domains |

Domain-specific types stay in their own packages.

---

## Interface Governance Principles

### 1. Minimize Shared Types

Only promote types to `razm:core` if they are used by 2+ domain interfaces. Domain-specific types live in their domain package.

### 2. Coarse-Grained Operations

Design functions to minimize cross-component calls. Pass large buffers, not individual pixels/samples.

### 3. Stream-First for Large Data

Use `input-stream`/`output-stream` (0.2) or `stream<T>` (0.3) for anything larger than a few MB. Never require loading an entire file into a `list<u8>`.

### 4. Resources for Stateful Operations

Encoders, decoders, and transformers should be resources with explicit lifecycle:
```
constructor → configure → process* → flush → [drop]
```

### 5. Version Everything

All packages use semver: `razm:core@0.1.0`. Breaking changes require a major version bump. Non-breaking additions use minor versions.

### 6. Design for 0.3 Migration

- Use streams where possible (maps to `stream<T>` in 0.3)
- Avoid complex pollable patterns (replaced by native async)
- Keep interfaces simple — 0.3 will simplify further

### 7. Plugin Interfaces are Contracts

Codec plugins implement well-defined interfaces (`razm:codec/encoder`, `razm:codec/decoder`). The interface IS the plugin API. No plugin registry, no discovery protocol — just WIT interface matching via `wasm-tools compose`.

### 8. Test Parity is Part of the Interface Contract

Every interface must have associated parity test specifications. An implementation is conformant only if it passes parity tests against reference outputs.

---

## WIT Versioning Strategy

### Semantic Versioning

```
razm:core@0.1.0    — Initial research/prototype
razm:core@0.2.0    — Breaking changes from prototype feedback
razm:core@1.0.0    — First stable release
razm:core@1.1.0    — Backwards-compatible additions
```

### Pre-1.0 Rules

- `0.x.0` releases may break interfaces freely
- Pin exact versions in dependent packages during pre-1.0

### Post-1.0 Rules

- Patch (`1.0.x`): Bug fixes only
- Minor (`1.x.0`): New functions/types, existing ones unchanged
- Major (`x.0.0`): Breaking changes (rename old interface, create new)

### WASI 0.2 → 0.3 Migration

Our interfaces will need a version bump when we adopt 0.3 async:

```wit
// 0.2 style (razm:image@0.1.0)
process: func(input: input-stream) -> result<output-stream, razm-error>;

// 0.3 style (razm:image@0.2.0)
process: async func(input: stream<u8>) -> result<stream<u8>, razm-error>;
```

Provide a 0.2→0.3 adapter component (polyfill) so consumers can upgrade at their own pace.

---

## Sources

- [WIT Reference — Component Model](https://component-model.bytecodealliance.org/design/wit.html)
- [Component Model Explainer](https://github.com/WebAssembly/component-model/blob/main/design/mvp/Explainer.md)
- [Composing Components — Fermyon](https://www.fermyon.com/blog/composing-components-with-spin-2)
- [WASI Current Status — eunomia](https://eunomia.dev/blog/2025/02/16/wasi-and-the-webassembly-component-model-current-status/)
- [WASI 0.3 Composable Concurrency](https://medium.com/wasm-radar/hypercharge-through-components-why-wasi-0-3-and-composable-concurrency-are-a-game-changer-0852e673830a)
