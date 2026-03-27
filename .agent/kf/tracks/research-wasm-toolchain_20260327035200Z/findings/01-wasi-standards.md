# WASI Standards Investigation

## Research Date: 2026-03-27

---

## WASI 0.2 (Preview 2) — STABLE

**Release:** January 25, 2024 (by Bytecode Alliance)
**Status:** Stable, production-ready

### Stable Interfaces

| Interface | Package | Purpose |
|-----------|---------|---------|
| I/O | `wasi:io` | Stream abstractions (input-stream, output-stream) |
| Clocks | `wasi:clocks` | Wall clock, monotonic clock |
| Random | `wasi:random` | Cryptographic random, insecure random |
| Filesystem | `wasi:filesystem` | Capability-based file access (preopened dirs) |
| Sockets | `wasi:sockets` | TCP/UDP networking |
| CLI | `wasi:cli` | Command-line world (stdin/stdout/stderr, env, args) |
| HTTP | `wasi:http` | HTTP proxy world (request/response model) |

### Worlds

- **wasi:cli/command** — For CLI tools with filesystem, sockets, clocks, random
- **wasi:http/proxy** — For HTTP handlers (request → response)

### Key Characteristics

- Built on the WebAssembly Component Model
- Uses WIT (WebAssembly Interface Type) for interface definitions
- Capability-based security — modules only access resources explicitly granted by host
- All interfaces defined in `.wit` files

---

## WASI 0.3 (Preview 3) — IN PREVIEW

**Status:** Preview available in Wasmtime 37+, completion ~February 2026
**RC Status:** WASIp3 RC landed in major runtimes as of November 2025

### What's New in 0.3

The headline feature is **native async support** in the Component Model:

- **`stream<T>`** — Typed streaming data between components
- **`future<T>`** — Single-value async completion
- Any component-level function can be implemented/called asynchronously
- Asynchrony implemented at the Canonical ABI level (not library-level)

### WASI 0.3 Refactors 0.2 Interfaces

The same 6 interfaces exist but are refactored to use native async:
- `wasi:io` streams become native `stream<T>` types
- Polling-based async (0.2) → native async (0.3)
- Backward compatibility: 0.2 can be polyfilled on top of 0.3

### Planned 0.3.x Point Releases (Two-Month Cadence)

| Feature | Status |
|---------|--------|
| Cancellation | Planned (language-pattern integration) |
| Tuple stream/future specialization | Planned |
| Stream optimization (zero-copy, forwarding) | Planned |
| Caller-supplied buffers | Planned |
| Cooperative threads | Planned |
| Preemptive threads | Later |

---

## WASI 1.0 — PLANNED

**Target:** Late 2026 or early 2027
**Scope:** Production-stable API freeze

---

## wasi:io Streams Interface — Deep Dive

**Package:** `wasi:io@0.2.8`
**Phase:** Phase 3 (stable, used as foundation by other WASI interfaces)

### input-stream Resource

| Method | Signature | Behavior |
|--------|-----------|----------|
| `read` | `(len: u64) → result<list<u8>, stream-error>` | Non-blocking, returns available bytes (may be 0) |
| `blocking-read` | `(len: u64) → result<list<u8>, stream-error>` | Blocks until at least 1 byte available |
| `skip` | `(len: u64) → result<u64, stream-error>` | Skip bytes without reading |
| `blocking-skip` | `(len: u64) → result<u64, stream-error>` | Blocking skip |
| `subscribe` | `() → pollable` | Get pollable for async waiting |

### output-stream Resource

| Method | Signature | Behavior |
|--------|-----------|----------|
| `check-write` | `() → result<u64, stream-error>` | How many bytes can be written now |
| `write` | `(contents: list<u8>) → result<_, stream-error>` | Non-blocking write |
| `blocking-write-and-flush` | `(contents: list<u8>) → result<_, stream-error>` | Up to ~4096 bytes per call |
| `flush` | `() → result<_, stream-error>` | Non-blocking flush |
| `blocking-flush` | `() → result<_, stream-error>` | Blocking flush |
| `splice` | `(src: borrow<input-stream>, len: u64) → result<u64, stream-error>` | Stream-to-stream transfer |
| `blocking-splice` | `(src: borrow<input-stream>, len: u64) → result<u64, stream-error>` | Blocking splice |

### stream-error Variant

```wit
variant stream-error {
    last-operation-failed(error),
    closed,
}
```

### Relevance to razm/core

**For large file processing (video, rosbag):**
- Host opens file → provides `input-stream` to WASM module
- Module reads in chunks via `read(65536)` (64KB) or larger
- Module writes to `output-stream` for output
- `splice` enables zero-copy forwarding between streams
- Module never sees file path — only the byte stream (security)

**Limitation in 0.2:** Polling-based async via `subscribe()` + `pollable` is clunky.
**Improvement in 0.3:** Native `stream<T>` types make this much more ergonomic.

**Buffer size note:** `blocking-write-and-flush` handles ~4096 bytes per call. For high-throughput media processing, the non-blocking `write` path with manual flush control is preferred.

---

## Recommendation for razm/core

### Target: WASI 0.2 now, design for 0.3 migration

**Rationale:**
1. WASI 0.2 is stable and has broad runtime support (Wasmtime, jco, etc.)
2. WASI 0.3 RC is available but not yet finalized — the async model is the key feature we want
3. The 0.2 → 0.3 migration path is well-defined (polyfill layer exists)
4. Our streaming use case will benefit significantly from 0.3's native `stream<T>`

**Strategy:**
- Build initial components targeting `wasm32-wasip2`
- Design WIT interfaces to be compatible with 0.3's async model
- Migrate to 0.3 when it stabilizes (expected H1 2026)
- Target WASI 1.0 for long-term stability (late 2026/early 2027)

---

## Sources

- [WASI 0.2 Launch — Bytecode Alliance](https://bytecodealliance.org/articles/WASI-0.2)
- [WASI Roadmap](https://wasi.dev/roadmap)
- [WASI Interfaces](https://wasi.dev/interfaces)
- [WASI 0.3 Native Async](https://byteiota.com/wasi-0-3-native-async-webassembly-gets-concurrent-i-o/)
- [WASIp3 RC Adoption](https://progosling.com/en/dev-digest/2026-01/wasip3-rc-adoption-checklist)
- [WASI I/O streams.wit](https://github.com/WebAssembly/wasi-io/blob/main/wit/streams.wit)
- [WebAssembly Updated Roadmap](https://bytecodealliance.org/articles/webassembly-the-updated-roadmap-for-developers)
- [WASI Current Status — eunomia](https://eunomia.dev/blog/2025/02/16/wasi-and-the-webassembly-component-model-current-status/)
- [State of WebAssembly 2025-2026](https://platform.uno/blog/the-state-of-webassembly-2025-2026/)
