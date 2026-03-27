# Plugin Distribution & Patent-Encumbered Codec Strategy

## Research Date: 2026-03-27

---

## 1. WASM Component Composition — How Plugins Work

### Build-Time Composition

The primary mechanism today. Multiple components are linked into a single binary:

```bash
# Core engine imports rasmcore:codec/encoder
# H.264 plugin exports rasmcore:codec/encoder
wasm-tools compose core-engine.wasm -d h264-plugin.wasm -o composed.wasm
```

**How it works:**
1. Core engine declares `import rasmcore:codec/encoder`
2. Plugin declares `export rasmcore:codec/encoder`
3. `wasm-tools compose` matches imports to exports by WIT interface
4. Result: single `.wasm` with both components, each with isolated linear memory
5. Mismatched WIT versions → runtime error

**Tools:**
- `wasm-tools compose` — lower-level, explicit composition
- `wac` (WebAssembly Composition) — higher-level, uses WAC language for composition rules

### Runtime Linking

Imports don't have to be fulfilled at build time — they can be resolved at instantiation:

```rust
// Host (wasmtime) loads core engine
let engine_component = Component::from_file(&engine, "core-engine.wasm")?;

// Host loads plugin separately
let plugin_component = Component::from_file(&engine, "h264-plugin.wasm")?;

// Host links plugin's exports to engine's imports at instantiation time
linker.instance("rasmcore:codec/encoder")?
    .func_wrap("encode-frame", |...| { /* delegate to plugin */ })?;
```

**Runtime linking is host-managed** — the host runtime (wasmtime, jco, etc.) decides which components to load and how to wire them together.

### wasmCloud Runtime Linking

wasmCloud takes this further with deployment manifests:

```yaml
components:
  - name: core-engine
    image: rasmcore/engine:0.1.0
  - name: h264-plugin
    image: rasmcore/h264:0.1.0
links:
  - source: core-engine
    target: h264-plugin
    interface: rasmcore:codec/encoder
```

The host coordinates links regardless of where components run (local, remote, distributed).

---

## 2. Plugin Discovery and Loading Patterns

### Pattern A: Explicit File Path (Simplest)

```bash
rasmcore-cli process video.mp4 --plugin ./h264-plugin.wasm
```

Host loads the plugin from the given path. No discovery needed.

### Pattern B: Plugin Directory

```
~/.rasmcore/plugins/
  h264-encoder.wasm
  hevc-decoder.wasm
  vp9-codec.wasm
```

Host scans a known directory for `.wasm` files, inspects their exports to determine what interfaces they provide.

### Pattern C: Registry Fetch

```bash
# Fetch plugin from OCI registry
wkg get rasmcore:codec-h264@0.1.0 -o h264-plugin.wasm

# Or in host code, fetch at startup
```

Host fetches plugins from a registry (OCI-based) at startup or on demand.

### Pattern D: Compose at Build Time (Recommended for Distribution)

```bash
# User composes their own binary with desired codecs
wasm-tools compose rasmcore-engine.wasm \
  -d h264-plugin.wasm \
  -d hevc-plugin.wasm \
  -o my-engine.wasm
```

User gets a single `.wasm` with exactly the codecs they need. No runtime discovery overhead.

### Recommendation for rasmcore

**Build-time composition as primary, runtime loading as advanced option.**

- Default: Users compose their desired plugins at build time → single `.wasm`
- Advanced: Host applications can load plugins at runtime via wasmtime API
- Both paths use the same WIT interfaces — the plugin doesn't know or care

---

## 3. Component Distribution via Registries

### OCI Registries (Current Standard)

The Bytecode Alliance uses OCI (Open Container Initiative) registries for WASM component distribution. This reuses existing container registry infrastructure.

**Tool:** `wkg` (wasm-pkg-tools)

```bash
# Publish a plugin
wkg publish rasmcore-h264-encoder.wasm --package rasmcore:codec-h264@0.1.0

# Fetch a plugin
wkg get rasmcore:codec-h264@0.1.0 -o h264-encoder.wasm

# Push to specific registry
wkg oci push ghcr.io/rasmcore/codec-h264:0.1.0 h264-encoder.wasm
```

### Namespace Configuration

```toml
# ~/.wkg/config.toml
default_registry = "ghcr.io"

[namespace_registries]
rasmcore = { registry = "rasmcore", metadata = {
  preferredProtocol = "oci",
  "oci" = { registry = "ghcr.io", namespacePrefix = "rasmcore/" }
}}
```

All `rasmcore:*` packages route to `ghcr.io/rasmcore/`.

### WIT Interface Publishing

WIT interfaces themselves can be published as packages:

```bash
# Publish the codec interface definition
wkg wit build --wit-dir wit/codec
wkg publish rasmcore-codec-interface.wasm --package rasmcore:codec@0.1.0

# Consumers fetch the interface to build against
wkg get --format wit rasmcore:codec@0.1.0 --output codec.wit
```

### Dependency Resolution

```bash
# Fetch all transitive deps, generate lockfile
wkg wit fetch
# → creates wit.lock pinning exact versions
# → populates deps/ directory
```

### Warg Status

The original Warg (WebAssembly Registry) protocol is **no longer actively developed** by Bytecode Alliance. Work has moved to OCI-based distribution via `wasm-pkg-tools`. This is good — OCI registries are battle-tested infrastructure (Docker Hub, ghcr.io, etc.).

---

## 4. Component Signing and Verification

### Current State

- `wkg` uses OS keyring for signing keys
- Package log entries are signed when publishing
- Registries accept packages based on signing keys
- **Package Transparency** (inspired by Certificate Transparency) is planned but not fully implemented

### What's Available

| Feature | Status |
|---------|--------|
| Signing on publish | Available (OS keyring) |
| Signature verification on fetch | Available |
| Package transparency logs | Planned |
| Content hashing (integrity) | Available (WASM binaries are hashable) |
| Revocation | Not standardized |

### For rasmcore

- Sign all published components with rasmcore project keys
- Consumers verify signatures on fetch
- WASM components are content-addressable (SHA-256 hash of binary)
- Consider publishing hash manifests alongside components

---

## 5. Codec Licensing Landscape

### Open / Royalty-Free Codecs

| Codec | License | rasmcore Strategy |
|-------|---------|-------------------|
| AV1 | BSD (Alliance for Open Media) | **Core module** — pure Rust (rav1e/rav1d) |
| VP9 | BSD (Google) | **Core module** — pure Rust reimplementation target |
| VP8 | BSD (Google) | **Core module** — pure Rust reimplementation target |
| Opus | BSD (Xiph/IETF) | **Core module** — pure Rust target |
| Vorbis | BSD (Xiph) | **Core module** — pure Rust (lewton decoder) |
| FLAC | BSD (Xiph) | **Core module** — pure Rust |
| Theora | BSD (Xiph) | Low priority — legacy codec |
| PNG/WebP/AVIF | Various open | **Core module** — via `image` crate |

### Patent-Encumbered Codecs — Sidecar Plugin

| Codec | Patent Pool(s) | rasmcore Strategy |
|-------|---------------|-------------------|
| H.264/AVC | MPEG-LA (Via Licensing) | **Plugin** — pure Rust reimplementation, sidecar distribution |
| H.265/HEVC | MPEG-LA, HEVC Advance, Velos Media, Technicolor | **Plugin** — highest legal complexity, pure Rust reimplementation |
| H.266/VVC | MPEG-LA + others | **Skip for now** — too new, patent landscape unclear |
| AAC | Via Licensing | **Plugin** — pure Rust reimplementation |
| MP3 (encode) | Patents expired (2017) | **Core module** — patents expired, safe to bundle |

### OpenH264 — Special Case

**Cisco's OpenH264** has a unique licensing model:

| Aspect | Detail |
|--------|--------|
| Source license | BSD 2-clause |
| Binary license | BSD + Cisco's MPEG-LA patent license (royalty-free) |
| Who pays royalties | Cisco (for their binaries only) |
| Third-party builds | **Must pay own MPEG-LA royalties** |
| Binary distribution rules | Must be separately downloaded, not pre-integrated |
| User control required | User must be able to enable/disable |

**Critical for rasmcore:** If we compile OpenH264 source to WASM ourselves, **Cisco does NOT cover the royalties** — we would need our own MPEG-LA license. Only Cisco's pre-built binaries carry their royalty coverage. Since Cisco doesn't provide WASM binaries, this path doesn't help us.

**Our strategy is better:** Pure Rust reimplementation of H.264 algorithms. The reimplementation is our own code, distributed as a separate sidecar plugin. Users who need H.264 consciously opt in by adding the plugin.

### H.265/HEVC — Most Complex

Four separate patent pools make H.265 the most legally hazardous codec:
1. MPEG-LA HEVC pool
2. HEVC Advance
3. Velos Media
4. Technicolor individual licensing

Even browser vendors (Chrome, Firefox) avoided H.265 due to this complexity. Our pure Rust reimplementation + sidecar distribution is the cleanest approach.

---

## 6. Proposed Plugin Architecture

### WIT Interface for Codec Plugins

```wit
package rasmcore:codec@0.1.0;

interface encoder {
    use rasmcore:core/types.{buffer, frame-info};

    resource encoder-instance {
        constructor(config: encoder-config);
        push-frame: func(data: buffer, info: frame-info) -> result<list<buffer>, codec-error>;
        flush: func() -> result<list<buffer>, codec-error>;
        get-config: func() -> encoder-config;
    }

    record encoder-config {
        codec-id: string,
        width: u32,
        height: u32,
        bitrate: u32,
        // ... codec-specific params as key-value
        params: list<tuple<string, string>>,
    }

    variant codec-error {
        unsupported-config(string),
        encode-failed(string),
        out-of-memory,
    }

    /// Discover what this plugin can encode
    supported-codecs: func() -> list<codec-info>;

    record codec-info {
        id: string,
        name: string,
        mime-type: string,
        patent-encumbered: bool,
    }
}

interface decoder {
    use rasmcore:core/types.{buffer, frame-info};

    resource decoder-instance {
        constructor(codec-id: string);
        push-packet: func(data: buffer) -> result<list<decoded-frame>, codec-error>;
        flush: func() -> result<list<decoded-frame>, codec-error>;
    }

    record decoded-frame {
        data: buffer,
        info: frame-info,
    }

    variant codec-error {
        unsupported-codec(string),
        decode-failed(string),
        out-of-memory,
    }

    supported-codecs: func() -> list<codec-info>;

    record codec-info {
        id: string,
        name: string,
    }
}
```

### World Definitions

```wit
// Core engine — imports codec interfaces (fulfilled by plugins)
world media-engine {
    import rasmcore:codec/encoder;
    import rasmcore:codec/decoder;
    import wasi:io/streams@0.2.8;
    export rasmcore:media/processor;
}

// Codec plugin — exports codec interfaces
world codec-plugin {
    export rasmcore:codec/encoder;
    export rasmcore:codec/decoder;
}
```

### Distribution Flow

```
1. rasmcore publishes:
   - rasmcore:codec@0.1.0 (WIT interface only — on OCI registry)
   - rasmcore:engine@0.1.0 (core engine .wasm — on OCI registry)
   - rasmcore:codec-av1@0.1.0 (AV1 plugin, open, bundled with core)
   - rasmcore:codec-h264@0.1.0 (H.264 plugin, sidecar, separate download)

2. User composes:
   wkg get rasmcore:engine@0.1.0
   wkg get rasmcore:codec-h264@0.1.0  # opt-in to patent-encumbered
   wasm-tools compose engine.wasm -d codec-h264.wasm -o my-engine.wasm

3. Result: single .wasm with exactly the codecs the user chose
```

---

## 7. Runtime Plugin Discovery Flow

For hosts that want runtime (not build-time) plugin loading:

```
1. Host starts, reads config for plugin directory or registry
2. Host scans for .wasm files matching rasmcore:codec/* pattern
3. For each plugin:
   a. Load component
   b. Call supported-codecs() to discover capabilities
   c. Register codec-id → plugin mapping
4. When user requests encode/decode:
   a. Look up codec-id in registry
   b. Instantiate the corresponding plugin
   c. Wire imports/exports
   d. Process data
```

This is the **advanced** path — most users will use build-time composition.

---

## 8. Recommendations Summary

| Decision | Recommendation |
|----------|---------------|
| Primary composition | Build-time via `wasm-tools compose` |
| Distribution registry | OCI (ghcr.io) via `wkg` |
| Plugin interface | WIT-defined encoder/decoder with `supported-codecs()` discovery |
| Open codecs | Bundle with core (AV1, VP9, FLAC, etc.) |
| Patent codecs | Separate sidecar plugins, opt-in download |
| H.264 implementation | Pure Rust reimplementation (NOT OpenH264 — their royalty coverage doesn't extend to third-party builds) |
| H.265 implementation | Pure Rust reimplementation, most complex legally |
| Component signing | Sign all published components, verify on fetch |
| Versioning | Semver via OCI tags (`rasmcore:codec-h264@0.1.0`) |

---

## Sources

- [wasm-tools compose — wasmCloud](https://wasmcloud.com/docs/concepts/linking-components/linking-at-build/)
- [WAC — GitHub](https://github.com/bytecodealliance/wac)
- [Building a Plugin System — DEV Community](https://dev.to/topheman/webassembly-component-model-building-a-plugin-system-58o0)
- [Distributing Components — Component Model docs](https://component-model.bytecodealliance.org/composing-and-distributing/distributing.html)
- [wasm-pkg-tools — GitHub](https://github.com/bytecodealliance/wasm-pkg-tools)
- [Warg — warg.io](https://warg.io/)
- [OpenH264 FAQ](https://www.openh264.org/faq.html)
- [OpenH264 — Wikipedia](https://en.wikipedia.org/wiki/OpenH264)
- [H.265 Patent Complexity — Wikipedia](https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding)
- [Codec Licensing and Streaming](https://www.streamingmedia.com/Articles/Post/Blog/Codec-Licensing-and-Web-Video-Streaming-161116.aspx)
