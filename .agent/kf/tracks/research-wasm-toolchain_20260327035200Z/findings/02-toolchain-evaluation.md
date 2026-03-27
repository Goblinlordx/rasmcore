# Toolchain Evaluation

## Research Date: 2026-03-27

---

## 1. cargo-component

**Repository:** [bytecodealliance/cargo-component](https://github.com/bytecodealliance/cargo-component)
**Status:** Experimental (API not stable)
**Purpose:** Cargo subcommand for creating WASM components from Rust

### Capabilities

| Command | Purpose |
|---------|---------|
| `cargo component new` | Scaffold a new component project |
| `cargo component add` | Add a WIT interface dependency |
| `cargo component build` | Build to WASM component |
| `cargo component update` | Update component lock file |
| `cargo component publish` | Publish to warg registry |

### Key Finding: Native Rust Target Alternative

**As of Rust 1.82**, the upstream `wasm32-wasip2` target produces components using plain `cargo build` — no `cargo-component` needed for WASI-only interfaces.

**When to use which:**

| Scenario | Tool |
|----------|------|
| Only WASI interfaces needed | `cargo build --target wasm32-wasip2` + `wasi` crate |
| Custom WIT interfaces (our case) | `cargo-component` required |
| Third-party WIT dependencies | `cargo-component` required |

### Implications for razm/core

Since we're defining custom WIT interfaces (image-processor, video-encoder, etc.), **we need cargo-component**. The native `wasm32-wasip2` target alone is insufficient.

**Risk:** cargo-component is explicitly "not currently stable" — upgrading may break builds. We should pin versions carefully.

---

## 2. wit-bindgen

**Repository:** [bytecodealliance/wit-bindgen](https://github.com/bytecodealliance/wit-bindgen)
**Purpose:** Generate language bindings from WIT interface definitions

### Language Support Matrix

#### Guest Languages (compiled TO WebAssembly)

| Language | Generator | Status | Notes |
|----------|-----------|--------|-------|
| **Rust** | `wit-bindgen-rust` | Mature | Primary language, best support |
| **C** | `wit-bindgen-c` | Mature | Low-level, manual memory mgmt |
| **Go (TinyGo)** | `wit-bindgen-go` (in go-modules) | Active | GC compatibility issues noted |
| **Java (TeaVM)** | `wit-bindgen-teavm-java` | Active | JVM-to-WASM via TeaVM |
| **C#** | `wit-bindgen-csharp` | Active | .NET support |

#### Host Languages (execute WebAssembly components)

| Language | Tool | Status | Notes |
|----------|------|--------|-------|
| **Rust** | `wasmtime` crate | Mature | Full Component Model support |
| **JavaScript/TS** | `jco` | Stable (1.0) | Transpiles components to ES modules |
| **Python** | `componentize-py` (guest only) | Active | v0.21.0 (Feb 2026), guest creation only |
| **Go** | wasmtime-go | **NO Component Model** | Core WASM only, no CM support |
| **Go** | wazero | **NO Component Model** | Pure Go, no CM support |

### Critical Gap: Go Host

**This is the most significant finding.** Neither wasmtime-go nor wazero support the Component Model for hosting.

- **wasmtime-go** (v42.0.1): Uses CGO to wrap Wasmtime's C API, but the C API lacks Component Model functions. Only core WASM modules supported.
- **wazero**: Pure Go runtime, no CGO. Supports WASM Core 1.0/2.0 only. No Component Model.
- **go-modules / wit-bindgen-go**: Generates *guest* bindings (Go compiled to WASM), not host bindings.

**Implication for razm/core:** Go cannot currently host WASM components. Options:
1. Wait for wasmtime-go to add Component Model C API bindings
2. Use wasmtime CLI as a subprocess from Go
3. Use jco to transpile components to JS/TS and call from Go via Node
4. Write a thin Go wrapper that calls wasmtime's Rust API via CGO
5. Contribute Component Model support to wasmtime-go (significant effort)

---

## 3. wasm-tools

**Repository:** [bytecodealliance/wasm-tools](https://github.com/bytecodealliance/wasm-tools)
**Purpose:** CLI and Rust libraries for WASM manipulation

### Key Subcommands

| Tool | Purpose | Maturity |
|------|---------|----------|
| `wasm-tools validate` | Validate WASM modules/components | Stable |
| `wasm-tools component new` | Create component from core module | Stable |
| `wasm-tools compose` | Compose multiple components | Stable |
| `wasm-tools print` | Print WAT from WASM binary | Stable |
| `wasm-tools parse` | Parse WAT to WASM binary | Stable |
| `wasm-tools dump` | Low-level binary dump | Stable |
| `wasm-tools strip` | Strip custom sections | Stable |
| `wasm-tools demangle` | Demangle function names | Stable |

### Component Composition

`wasm-tools compose` is the critical tool for our plugin architecture:
- Links components by matching imports/exports
- Enables build-time composition of modules
- Supports virtual component adapters

### Validation

Implements all standardized proposals (Stage 4+) with per-proposal feature flags. Critical for CI/CD — can validate that our components are well-formed before distribution.

---

## Summary: Toolchain Maturity Assessment

| Tool | Maturity | Risk for razm/core |
|------|----------|---------------------|
| cargo-component | Experimental | Medium — API instability, but required for custom WIT |
| wit-bindgen (Rust guest) | Mature | Low — well-supported primary path |
| wit-bindgen (Go guest via TinyGo) | Active | Medium — GC issues noted |
| wasm-tools | Stable | Low — solid CLI tooling |
| wasmtime (Rust host) | Mature | Low — production-ready |
| jco (JS/TS host) | Stable (1.0) | Low — good for TS consumers |
| componentize-py | Active (v0.21) | Medium — fast-moving, guest only |
| **Go host (any)** | **Missing** | **HIGH — No Component Model host for Go** |

---

## Sources

- [cargo-component GitHub](https://github.com/bytecodealliance/cargo-component)
- [wit-bindgen GitHub](https://github.com/bytecodealliance/wit-bindgen)
- [wasm-tools GitHub](https://github.com/bytecodealliance/wasm-tools)
- [go-modules GitHub](https://github.com/bytecodealliance/go-modules)
- [wasmtime-go GitHub](https://github.com/bytecodealliance/wasmtime-go)
- [wazero](https://wazero.io/)
- [jco 1.0 Announcement](https://bytecodealliance.org/articles/jco-1.0)
- [componentize-py Releases](https://github.com/bytecodealliance/componentize-py/releases)
