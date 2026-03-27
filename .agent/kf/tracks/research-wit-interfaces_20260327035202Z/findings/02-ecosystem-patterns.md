# Ecosystem WIT Patterns

## Research Date: 2026-03-27

---

## 1. Fermyon Spin — Component Composition Patterns

### How Spin Defines Worlds

Spin components implement the `wasi:http/incoming-handler` world. Custom components define their own worlds with explicit imports/exports:

```wit
package middleware:http-auth;

world wasi-http-import {
    import wasi:http/incoming-handler@0.2.0;
}
```

### Composition Pattern

1. **Component A** exports an interface (e.g., `wasi:http/incoming-handler`)
2. **Component B** imports that same interface
3. `wasm-tools compose` links them: B's import is satisfied by A's export
4. Result: single composed component, each with isolated linear memory

```bash
wasm-tools compose auth.wasm -d business-logic.wasm -o service.wasm
```

### Key Design Decision

Spin uses **WIT interface matching** for composition — not custom plugin discovery. If two components agree on a WIT interface, they can be composed. This is exactly the pattern we need for razm/core plugins.

### Dependency Resolution

```toml
[package.metadata.component.target.dependencies]
"wasi:http" = { path = "wit/deps/http" }
"wasi:io" = { path = "wit/deps/io" }
```

WIT dependencies are specified as file paths in Cargo.toml. `cargo-component` resolves them during build.

---

## 2. Fastly Compute — Transitional Pattern

Fastly currently uses **witx** (precursor to WIT) for their ABI definitions. They are transitioning to full WIT/Component Model.

**Relevant insight:** Even major production platforms are still migrating. Our choice to start fresh with WIT is an advantage — no legacy witx to carry.

---

## 3. Common Patterns Across Projects

### Pattern: Shared Types Interface

Every project that defines custom WIT interfaces uses a dedicated `types` interface:

```wit
package razm:core@0.1.0;

interface types {
    type buffer = list<u8>;

    record image-metadata {
        width: u32,
        height: u32,
        format: pixel-format,
        color-space: color-space,
    }

    enum pixel-format {
        rgb8,
        rgba8,
        gray8,
        gray16,
    }

    enum color-space {
        srgb,
        linear,
        display-p3,
    }
}
```

Other interfaces `use types.{buffer, image-metadata};` to share these definitions.

### Pattern: Resource for Stateful Processing

Long-running or stateful operations are modeled as resources:

```wit
resource encoder {
    constructor(config: encoder-config);
    push-frame: func(frame: borrow<frame>) -> result<_, encode-error>;
    flush: func() -> result<list<u8>, encode-error>;
}
```

This maps well to our video encoder/decoder use case.

### Pattern: Worlds as Role Definitions

Worlds define the "role" a component plays in the ecosystem:

```wit
// A component that processes images
world image-processor {
    import wasi:io/streams@0.2.8;
    import razm:core/types@0.1.0;
    export razm:image/processor@0.1.0;
}

// A component that provides an encoder plugin
world codec-plugin {
    import razm:core/types@0.1.0;
    export razm:video/encoder@0.1.0;
}
```

### Pattern: Import WASI, Export Domain

Components import WASI system interfaces (I/O, filesystem, clocks) and export domain-specific interfaces. This is the standard separation:

- **Imports**: System capabilities the component needs (host provides)
- **Exports**: Domain functionality the component offers (host calls)

---

## 4. Composition Model for Plugin Architecture

### Build-Time Composition (Current — WASI 0.2)

```
[core-engine.wasm] ←imports encoder→ [h264-plugin.wasm]
                          ↓
           wasm-tools compose
                          ↓
              [composed-engine.wasm]
```

- `core-engine` imports `razm:video/encoder`
- `h264-plugin` exports `razm:video/encoder`
- `wasm-tools compose` links them into a single component
- Result has isolated memory per sub-component

### Runtime Composition (Future — WASI 0.3+)

With 0.3 and runtime instantiation support, the host could:
1. Load `core-engine.wasm`
2. Discover `h264-plugin.wasm` at runtime
3. Link them dynamically via matching imports/exports

This is not fully standardized yet but is the direction the ecosystem is heading.

### Assessment for razm/core

**Build-time composition works today** and is well-supported by `wasm-tools compose`. This is sufficient for our plugin model — users compose their needed codecs at build time. Runtime discovery is a future enhancement.

---

## Sources

- [Composing Components with Spin 2.0 — Fermyon](https://www.fermyon.com/blog/composing-components-with-spin-2)
- [Introducing Spin 3.0 — Fermyon](https://www.fermyon.com/blog/introducing-spin-v3)
- [Component Reuse with Spin — Fermyon](https://www.fermyon.com/blog/component-reuse)
- [Fastly compute-at-edge-abi — GitHub](https://github.com/fastly/compute-at-edge-abi)
- [WASM Component Model: LEGO Bricks](https://www.javacodegeeks.com/2026/02/the-wasm-component-model-software-from-lego-bricks.html)
