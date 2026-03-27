# Host SDK Assessment

## Research Date: 2026-03-27

---

## 1. Wasmtime (Rust) — REFERENCE IMPLEMENTATION

**Crate:** `wasmtime` (with `component-model` feature, enabled by default)
**Latest:** Actively tracking WASI 0.2.1+ and experimental 0.3
**Status:** Production-ready, mature

### Component Model API

```rust
// The embedding API mirrors core wasm but adds component concepts
wasmtime::component::Component  // Compiled component (like Module for core)
wasmtime::component::Linker     // Define host functions for components
wasmtime::component::Instance   // Instantiated component
```

- Maps all WIT types to Rust automatically
- Generates traits for embedders to implement
- Supports async host calls (experimental, via Rust async/await)
- Full WASI 0.2.1 support
- Experimental WASI 0.3 support (Wasmtime 37+)

### LTS Policy

- LTS release every 12 versions
- Up to 24 months of security patches
- Indicates commitment to production stability

### Assessment for razm/core

**Verdict: STRONG GO.** Wasmtime Rust embedding is the gold-standard host. If we build our own host runtime in Rust, this is the clear choice. All WIT features work, async is coming, and it's the reference implementation.

---

## 2. jco (JavaScript/TypeScript) — STABLE

**Package:** `@bytecodealliance/jco` (npm)
**Version:** 1.0 (stable)
**Purpose:** JavaScript toolchain for WebAssembly Components

### Capabilities

| Feature | Status |
|---------|--------|
| Transpile WASM component → ES modules | Stable |
| Run components in Node.js | Stable |
| Run components in browser | Stable |
| TypeScript type generation | Stable |
| ComponentizeJS (JS → WASM component) | Stable |
| WASI 0.2 runtime for Node.js | Stable |

### How It Works

1. Takes a `.wasm` component binary
2. Transpiles it to standard ES module(s)
3. Generates TypeScript declarations
4. The ES module can be imported by any JS/TS project

### Key Strengths

- **No runtime dependency** — transpiled output is plain JS
- **Browser support** — via `@bytecodealliance/jco/component` import
- **npm integration** — components can be published as npm packages
- **TypeScript-first** — generates `.d.ts` files from WIT

### Assessment for razm/core

**Verdict: STRONG GO.** jco 1.0 is the best path for TypeScript consumers. Our Rust WASM components can be transpiled to npm packages with full TypeScript types. This is production-ready today.

---

## 3. componentize-py (Python) — ACTIVE, GUEST-ONLY

**Package:** `componentize-py` (PyPI)
**Version:** v0.21.0 (February 2026)
**Purpose:** Convert Python applications to WASM components

### Current State

- Actively developed (monthly releases throughout 2025-2026)
- Creates WASM components FROM Python code
- Uses CPython compiled to WASM as the runtime
- Supports WASI 0.2 interfaces

### Limitation: Guest Only

componentize-py creates Python-as-WASM components (guest). It does NOT provide a Python host runtime for executing WASM components.

**For Python hosting,** you would need:
- `wasmtime` Python bindings (exists but Component Model support unclear)
- Or: transpile component to Python-callable format via jco then call from Python

### Assessment for razm/core

**Verdict: PARTIAL.** Python can create WASM components (useful if we want Python-implemented plugins). But Python cannot easily HOST our Rust WASM components today. The primary use case — importing our image/video/data modules from Python — would likely go through:
1. A native Python extension built from the Rust code directly (via PyO3/maturin)
2. Or: wasmtime Python bindings (needs Component Model verification)

**Risk: Medium.** Python host story is the weakest of our three target languages.

---

## 4. Go Host — CRITICAL GAP

### wasmtime-go

**Version:** v42.0.1
**Architecture:** CGO wrapper around Wasmtime's C API
**Component Model:** NOT SUPPORTED

The Wasmtime C API does not expose Component Model functions. Since wasmtime-go wraps the C API, it inherits this limitation. Only core WASM modules can be hosted.

### wazero

**Version:** Stable (1.0+)
**Architecture:** Pure Go, zero dependencies
**Component Model:** NOT SUPPORTED

wazero implements WASM Core 1.0/2.0 spec only. No Component Model support planned in the near term.

### go-modules / wit-bindgen-go

This is for Go-as-GUEST (TinyGo compiled to WASM), NOT Go-as-HOST.
- v0.7.0 (May 2025)
- Known GC compatibility issues with standard Go (works better with TinyGo)
- Active development, targeting WASIp3

### Workaround Options

| Option | Feasibility | Effort | Performance |
|--------|------------|--------|-------------|
| Wait for wasmtime C API Component Model | Unknown timeline | None (wait) | Native |
| Subprocess: `wasmtime` CLI | Easy | Low | Process overhead |
| Use jco to transpile, call from Go via Node bridge | Hacky | Medium | JS overhead |
| Rust FFI: embed wasmtime Rust via CGO | Complex | High | Near-native |
| Contribute CM to wasmtime-go | Very high effort | Very High | Native |
| Skip WASM for Go — provide native Rust lib via CGO | Pragmatic | Medium | Native |

### Assessment for razm/core

**Verdict: HIGH RISK.** Go is the weakest link in our host story. The most pragmatic near-term approach may be to provide Go users with a native Rust library (via CGO or a C API) alongside the WASM components, rather than requiring Go to host WASM components.

**Alternative:** If we expose our modules via a CLI or HTTP API (using wasmtime Rust host internally), Go applications can consume them without needing a Go WASM host.

---

## Cross-Language Summary

| Host Language | Component Model | WASI 0.2 | Method | Production Ready |
|---------------|-----------------|----------|--------|------------------|
| **Rust** | Full | Full | wasmtime crate | YES |
| **TypeScript** | Full | Full | jco 1.0 transpile | YES |
| **Python** | Partial (guest) | Partial | componentize-py / wasmtime-py | PARTIAL |
| **Go** | NONE | Core only | N/A | NO |

---

## Sources

- [wasmtime::component Rust API](https://docs.wasmtime.dev/api/wasmtime/component/)
- [jco 1.0 — Bytecode Alliance](https://bytecodealliance.org/articles/jco-1.0)
- [jco GitHub](https://github.com/bytecodealliance/jco)
- [componentize-py GitHub](https://github.com/bytecodealliance/componentize-py)
- [wasmtime-go GitHub](https://github.com/bytecodealliance/wasmtime-go)
- [wazero](https://wazero.io/)
- [go-modules GitHub](https://github.com/bytecodealliance/go-modules)
- [Arcjet: WebAssembly from Go](https://blog.arcjet.com/webassembly-on-the-server-compiling-rust-to-wasm-and-executing-it-from-go/)
