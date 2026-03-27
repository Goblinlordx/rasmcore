# Image Processing Crate Evaluation

## Research Date: 2026-03-27

---

## 1. `image` Crate (image-rs)

**Repository:** [image-rs/image](https://github.com/image-rs/image)
**Status:** Mature, widely used (the standard Rust image library)
**Pure Rust:** Mostly yes, with caveats

### Supported Formats

| Format | Decode | Encode | Pure Rust | Notes |
|--------|--------|--------|-----------|-------|
| PNG | Yes | Yes | Yes | |
| JPEG | Yes | Yes | Yes | |
| GIF | Yes | Yes | Yes | |
| BMP | Yes | Yes | Yes | |
| TIFF | Yes | Yes | Yes | |
| WebP | Yes | Yes | Yes | |
| AVIF | Yes | Yes | Yes (default) | `avif-native` feature adds C deps (dav1d, mp4parse) |
| ICO | Yes | Yes | Yes | |
| PNM | Yes | Yes | Yes | |
| TGA | Yes | Yes | Yes | |
| QOI | Yes | Yes | Yes | |
| HDR | Yes | Yes | Yes | |
| EXR | Yes | Yes | Yes | OpenEXR format |
| Farbfeld | Yes | Yes | Yes | |

### WASM Compilation

**Works with caveats:**
- Default features include `rayon` (multithreading) — **incompatible with WASM**
- Must use `default-features = false` and explicitly enable format features
- Example: `image = { version = "0.25", default-features = false, features = ["png", "jpeg", "webp"] }`
- No known issues with `wasm32-wasip2` target when configured correctly

### Assessment

**STRONG CANDIDATE.** The `image` crate covers all common formats in pure Rust. The WASM issue is just a feature flag configuration. This should be our primary image format handling library.

---

## 2. OxiMedia

**Repository:** [cool-japan/oximedia](https://github.com/cool-japan/oximedia)
**Version:** v0.1.2 (2026-03-17)
**Status:** CAUTION — significant concerns

### What It Claims

- 106 crates, ~2.16M lines of Rust
- Pure Rust reconstruction of FFmpeg + OpenCV
- Patent-free (no H.264/H.265)
- Apache 2.0 license
- Supports: AV1, VP9, VP8, Theora, Opus, Vorbis, FLAC, PNG, WebP, AVIF
- Claims `wasm32-unknown-unknown` support

### Community Concerns (Hacker News)

**SERIOUS RED FLAGS identified by the community:**

1. **AI-generated code suspected** — Reviewers noted patterns consistent with LLM output without disclosure. Specific crates (e.g., `oximedia-gaming` NVENC support) found to be "a no-op" — code that compiles but doesn't actually work.

2. **Single-commit repository** — All ~2M lines added in one commit. No development history, no way to evaluate code evolution or quality.

3. **No benchmarks** — Zero performance comparisons against FFmpeg, rav1e, dav1d, or any other tool. Developer promised benchmarks "coming soon."

4. **"Production-ready" walked back** — After pushback, developer changed claim to "API-stable." Big difference.

5. **Impossible to audit** — "Nobody is gonna review 1M lines" — the sheer scale makes quality verification impractical.

### Assessment

**DO NOT DEPEND ON for razm/core.** OxiMedia's claims are unverified. The AI-generation concerns, lack of benchmarks, and single-commit history mean we cannot trust it for a security-focused project. Individual, well-established crates (image, rav1e, rav1d) are far safer choices.

**Watch from a distance** — if OxiMedia matures with proper development history, benchmarks, and community review over the next 6-12 months, it could become relevant.

---

## 3. Photon (photon-rs)

**Repository:** [silvia-odwyer/photon](https://github.com/silvia-odwyer/photon)
**Status:** Active, WASM-first design
**Pure Rust:** 100%

### Capabilities

- 96 image processing functions
- Hue rotation, sharpening, brightness, saturation
- Convolutions: Sobel, blur, Laplace, edge detection
- Channel manipulation: RGB adjustment, swapping, removal
- Transforms: resize, crop, rotate, flip
- Monochrome: duotone, greyscale, threshold, sepia
- Color spaces: HSL, LCh, sRGB

### WASM Support

- **Designed for WASM from the start**
- Works in browser (Chrome, Firefox, Safari, Edge)
- Node.js support via npm package
- 4-10x faster than JavaScript image processing
- Pure Rust, no C dependencies

### Assessment

**GOOD COMPLEMENT to `image` crate.** Photon handles image manipulation/effects while `image` handles format I/O. Photon's WASM-first design makes it ideal for our use case.

---

## 4. resvg + tiny-skia

**Repository:** [linebender/resvg](https://github.com/linebender/resvg), [linebender/tiny-skia](https://github.com/linebender/tiny-skia)
**Status:** Mature, actively maintained
**Pure Rust:** Yes (no non-Rust code in final binary)

### Capabilities

- **resvg:** Complete SVG rendering (goal: full SVG spec support)
- **tiny-skia:** Minimal CPU-only 2D rendering (Skia subset)
- **resvg-wasm:** Browser WASM bindings for SVG→PNG

### WASM Support

- Guaranteed to work everywhere Rust compiles, including WASM
- resvg-wasm package exists specifically for browser use
- Pure Rust — no system dependencies

### Assessment

**USEFUL for SVG support.** If razm/core needs SVG rendering, resvg is the clear choice. It's a specialized tool, not a general image library.

---

## 5. Other Notable Crates

| Crate | Purpose | Pure Rust | WASM | Notes |
|-------|---------|-----------|------|-------|
| `zune-image` | Fast image decoding | Yes | Yes | Focus on decoder performance |
| `ril` | High-level image lib | Yes | Partial | Simpler API than `image` |
| `imageproc` | Image processing ops | Yes | Yes | Built on `image` crate |

---

## Recommendations for razm/core Image Module

### Primary Stack

| Layer | Crate | Role |
|-------|-------|------|
| Format I/O | `image` (with `default-features = false`) | Decode/encode all standard formats |
| Processing | `photon-rs` | Effects, filters, transforms |
| SVG | `resvg` | SVG rendering (if needed) |
| 2D Rendering | `tiny-skia` | Low-level rasterization (if needed) |

### Configuration for WASM

```toml
[dependencies]
image = { version = "0.25", default-features = false, features = [
    "png", "jpeg", "gif", "webp", "bmp", "tiff", "avif", "ico", "qoi"
] }
photon-rs = "0.3"
```

### Gaps

- **No pure-Rust RAW format support** (camera RAW files) — would need custom implementation
- **HEIF/HEIC** — patent-encumbered, no pure-Rust implementation (plugin candidate)

---

## Sources

- [image crate — GitHub](https://github.com/image-rs/image)
- [OxiMedia — GitHub](https://github.com/cool-japan/oximedia)
- [OxiMedia HN Discussion](https://news.ycombinator.com/item?id=47302515)
- [Photon — GitHub](https://github.com/silvia-odwyer/photon)
- [resvg — GitHub](https://github.com/linebender/resvg)
- [tiny-skia — GitHub](https://github.com/linebender/tiny-skia)
