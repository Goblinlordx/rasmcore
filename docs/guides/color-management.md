# Color Management in rasmcore

A comprehensive guide to color spaces, transforms, and the ACES pipeline — how professional color management works in the industry and how rasmcore implements it.

## Fundamentals

### What is a Color Space?

A color space defines how RGB numbers map to real-world colors. It has two independent properties:

**Primaries** — which real colors R, G, B represent. This defines the *gamut* (range of reproducible colors). Different sets of primaries cover different portions of the visible spectrum.

**Transfer function** — how brightness values are encoded. Linear means proportional to light intensity (double the value = double the photons). Gamma, log, and PQ are nonlinear encodings that distribute precision differently across the brightness range.

| Color Space | Primaries | Transfer Function | Use |
|---|---|---|---|
| sRGB | Rec.709 | sRGB gamma (~2.2) | Web, consumer displays |
| Linear sRGB | Rec.709 | Linear | Internal computation |
| Rec.709 | Rec.709 | BT.1886 EOTF | HD broadcast |
| Rec.2020 | Rec.2020 (wide) | Various (HLG, PQ) | HDR/UHD broadcast |
| Display P3 | DCI-P3 (wide) | sRGB gamma | Modern Apple/Android displays |
| ACEScg | AP1 (wide) | Linear | VFX compositing |
| ACEScct | AP1 | Log (with toe) | Color grading |
| ACES2065-1 | AP0 (very wide) | Linear | Archival interchange |

### What is a Transfer Function?

The transfer function (also called OETF/EOTF) is the curve mapping between linear light and encoded signal values.

**Why not just use linear?** Because human vision is roughly logarithmic — we're much more sensitive to differences in dark tones than bright tones. An 8-bit linear encoding wastes most of its 256 levels on bright values we can barely distinguish, while dark values get harsh banding. Nonlinear encoding distributes precision to match human perception.

Common transfer functions:

| Name | Type | Used by | Character |
|---|---|---|---|
| sRGB gamma | Display EOTF | sRGB displays | ~2.2 power with linear toe segment |
| BT.1886 | Display EOTF | Rec.709 broadcast | Pure 2.4 power |
| PQ (ST.2084) | Display EOTF | HDR (Dolby Vision, HDR10) | Perceptual quantizer, 10000 nits |
| HLG | Scene OETF | HDR broadcast | Backwards-compatible with SDR |
| ARRI LogC | Camera OETF | ARRI Alexa cameras | Logarithmic, maximizes sensor DR |
| Sony S-Log3 | Camera OETF | Sony cameras | Logarithmic |
| ACEScct | Working space | ACES grading | Log with toe, "musical" sliders |

### What are Luminance Coefficients?

When calculating perceived brightness of a color, you weight R, G, B differently because the eye is most sensitive to green, moderately to red, and least to blue. These weights are the *luminance coefficients*, derived mathematically from the color space's primaries.

rasmcore derives these automatically from the working color space via `ColorSpace::luma_coefficients()`. Filters that compute luminance (blend modes, saturation, luminance masks) use the correct coefficients for whatever space the data is in.

| Color Space Primaries | R weight | G weight | B weight |
|---|---|---|---|
| Rec.709 / sRGB | 0.2126 | 0.7152 | 0.0722 |
| Rec.2020 | 0.2627 | 0.6780 | 0.0593 |
| AP1 (ACEScg/ACEScct) | 0.2722 | 0.6741 | 0.0537 |
| Display P3 | 0.2290 | 0.6917 | 0.0793 |

## The ACES Pipeline

### What is ACES?

ACES (Academy Color Encoding System) is the industry-standard color management framework for cinema and professional imaging. Developed by the Academy of Motion Picture Arts and Sciences (the Oscars people), it provides:

1. A common interchange format (ACES2065-1)
2. Standard working spaces (ACEScg for compositing, ACEScct for grading)
3. Camera-to-ACES input transforms (IDTs) for every major camera
4. ACES-to-display output transforms for every display standard
5. A defined rendering intent (how scene light becomes viewable images)

### The ACES Viewing Pipeline

```
Camera → IDT → ACES2065-1 → CSC → ACEScg → LMT → CSC → ACEScct → grading
   │                                                         │
   │                            ┌─────────────────────────────┘
   │                            ▼
   │                         ACEScg → Output Transform → Display
   │                                       │
   │                            ┌──────────┴──────────┐
   │                            │    ACES 1.x:        │
   │                            │    RRT → OCES → ODT │
   │                            │    ACES 2.0:        │
   │                            │    Single OT        │
   │                            └─────────────────────┘
```

### IDT — Input Device Transform

**What:** Converts camera-native footage to ACES2065-1 (AP0 linear).

**How it works:** Two steps:
1. **Linearize** — undo the camera's log/gamma encoding (transfer function → linear)
2. **Matrix** — 3x3 transform from camera primaries to ACES AP0 primaries

**Distribution:** CTL files (ACES reference) or CLF files (practical interchange). One file per camera model, and often parameterized by exposure index (EI) and color temperature (CCT).

**Key property: one-directional.** You cannot invert an IDT to reconstruct camera-native values because the camera's spectral response doesn't perfectly match human vision. The conversion is mathematically invertible, but the inverse doesn't produce valid camera code values.

**Not ACES-versioned.** IDTs are camera-specific. The "v4" in "ARRI LogC v4" is ARRI's version, not an ACES version. An IDT from ACES 1.0 works with ACES 2.0 — the camera-to-AP0 math doesn't change.

Example IDTs:
- `IDT.ARRI.Alexa-v2-logC-EI800` — ARRI Alexa, LogC encoding, EI 800
- `IDT.Sony.SLog3_SGamut3` — Sony, S-Log3 curve, S-Gamut3 primaries
- `IDT.RED.REDWideGamutRGB_Log3G10` — RED, Log3G10 curve, REDWideGamut

### ACEScg and ACEScct — Working Spaces

**ACEScg (AP1 primaries, linear):** The standard compositing working space. "cg" = computer graphics. All spatial operations (blur, resize, composite, blend) should happen here because the math requires linear light.

**ACEScct (AP1 primaries, log with toe):** The standard grading working space. "cct" = color correction with toe. The log encoding makes slider adjustments perceptually uniform — a "+0.1" brightness change "feels" the same in shadows and highlights.

**Key facts:**
- Same primaries (AP1) — converting between them is just a 1D per-channel curve, lossless
- No versions — these are fixed mathematical definitions, unchanged since their introduction
- The choice of which to work in depends on the operation type, not user preference

### LMT — Look Modification Transform

**What:** A creative "look" applied in the ACES working space. Film emulation, stylistic grades, show-specific color treatments.

**How it works:** Can be anything — a 3D LUT, a CDL (slope/offset/power), a complex multi-node CLF process list. The only constraint is input and output are both ACES.

**Key property: one-directional.** LMTs are creative transforms, not mathematically invertible color space definitions. A film emulation LUT deliberately destroys information (compresses gamut, crushes blacks). There's no "undo" beyond removing it.

**Distribution:** .cube files (3D LUTs), .clf files (CLF process lists), .cdl files (ASC CDL grades).

### Output Transform — RRT + ODT

This is where **ACES versions matter**.

**RRT (Reference Rendering Transform):** The "ACES look" — maps unbounded scene-referred values to a viewable range. Includes tone mapping, desaturation at extremes, and color appearance adjustments. The same regardless of display.

**ODT (Output Device Transform):** Adapts the RRT output for a specific display — applies the display's EOTF (gamma, PQ), maps to the display's gamut, adjusts for viewing conditions.

**In ACES 1.x (versions 1.0 through 1.3):**
- RRT and ODT are separate CTL files
- Fixed segmented-spline tonescale
- Per-channel processing in RGB space
- Separate ODT file per display (sRGB, Rec.709, P3, etc.)
- Known issue: poor gamut mapping (blue neon lights turn purple)

**In ACES 2.0 (2024):**
- RRT + ODT replaced by a single parametric Output Transform
- Works in perceptual JMh space (based on CAM16 color appearance model)
- Parametric: `outputTransform(peakLuminance=100, primaries=bt709)`
- Same function for SDR and HDR (just different parameters)
- Much better gamut mapping
- No more separate RRT and ODT files

**Compatibility:** IDTs from any ACES version work with ACES 2.0 (camera math doesn't change). But the Output Transform results are visibly different between 1.x and 2.0 — you need to specify which you're using.

### Version Summary

| Component | Versioned? | Notes |
|---|---|---|
| IDT | No (camera-versioned, not ACES-versioned) | ARRI LogC "v4" is ARRI's version |
| ACES2065-1 | No | Fixed since SMPTE ST 2065-1 |
| ACEScg | No | Fixed since S-2014-004 |
| ACEScct | No | Fixed since S-2016-001 |
| LMT | No (user-created) | Your creative grade |
| Output Transform | **Yes** | ACES 1.3 vs 2.0 produce different results |

## File Formats

### CTL (Color Transform Language)

**Spec:** SMPTE ST 2065-3

The canonical format for ACES transforms. Plain text, C-like procedural code. This is what the Academy publishes in the aces-dev GitHub repository.

Used for: reference implementations of IDTs, RRT, ODTs.

Not commonly consumed directly by applications — most tools use CLF or OCIO configs instead.

### CLF (Common LUT Format)

**Spec:** SMPTE ST 2065-4 (also Academy/ASC Common LUT Format)

XML-based format with a fixed set of process node types. The practical interchange format for color transforms.

Process node types:
- `LUT1D` — 1D lookup table
- `LUT3D` — 3D lookup table
- `Matrix` — 3x3 (or 3x4) matrix
- `Range` — clamp/scale values
- `Log` — logarithmic transfer function (with `cameraLogToLin` style for camera curves)
- `Exponent` — power function
- `ASC_CDL` — ASC Color Decision List

A CLF file is an ordered list of these nodes — the transform is the sequential application of all nodes.

### CDL (ASC Color Decision List)

**Spec:** ASC CDL (American Society of Cinematographers)

The universal on-set grading format. Extremely simple — per-channel slope, offset, power, plus global saturation:

```
out = clamp((in * slope + offset) ^ power)
out = luma + saturation * (out - luma)
```

Distributed as `.cdl`, `.cc`, or `.ccc` XML files. Every camera department generates CDLs on set for dailies.

### .cube (Iridas/Resolve 3D LUT)

A simple text format for 1D and 3D LUTs. De facto standard for LUT interchange because of its simplicity. Not an official standard — originated from Iridas (now part of Adobe), popularized by DaVinci Resolve.

## How rasmcore Implements This

### The Color Transform Resource

Color transforms follow the same resource pattern as Fonts — register once, reference by ID:

```javascript
// Built-in: get reference (lazy-parsed from embedded CLF on first access)
const idt = pipeline.getTransform("idt-srgb");
const ot  = pipeline.getTransform("ot-rec709");

// Custom: register from file data
const lmt = pipeline.registerTransform(cubeFileBytes, "cube");
const customIdt = pipeline.registerTransform(clfFileBytes, "clf");

// Apply in pipeline — each transform knows its own from/to
pipeline.read(image)
  .applyIdt(idt)              // sRGB → ACEScg
  .applyLmt(lmt)              // creative look
  .applyOutputTransform(ot)   // ACEScg → Rec.709
```

### Available Built-in Transforms

| Name | Kind | Source → Target | Vendor |
|---|---|---|---|
| `idt-srgb` | IDT | sRGB → ACEScg | Academy |
| `idt-rec709` | IDT | Rec.709 → ACEScg | Academy |
| `idt-rec2020` | IDT | Rec.2020 → ACEScg | Academy |
| `idt-p3` | IDT | Display P3 → ACEScg | Academy |
| `ot-srgb` | Output Transform | ACEScg → sRGB | Academy |
| `ot-rec709` | Output Transform | ACEScg → Rec.709 | Academy |
| `ot-rec2020` | Output Transform | ACEScg → Rec.2020 | Academy |
| `ot-p3` | Output Transform | ACEScg → Display P3 | Academy |
| `csc-acescg-to-cct` | CSC | ACEScg → ACEScct | Academy |
| `csc-acescct-to-cg` | CSC | ACEScct → ACEScg | Academy |

Camera-specific IDTs (ARRI, Sony, RED) will be added in a future track as actual CLF files from vendor specifications.

### Color-Space-Aware Filters

Filters that depend on luminance (blend modes, saturation, vibrance, luminance masks) automatically use the correct luminance coefficients for the working color space. This is handled by `ColorSpace::luma_coefficients()` — filters never hardcode BT.709 or NTSC weights.

Blend mode formulas follow ISO 32000-2:2020 (PDF specification). Non-separable modes (hue, saturation, color, luminosity) use the W3C/PDF SetLum/SetSat/ClipColor helpers with working-space-derived coefficients.

### Working Color Space

The pipeline supports setting a default working color space:

```javascript
pipeline.setWorkingColorSpace("acescg");
```

When set, the pipeline auto-inserts color space conversions at source boundaries (after decode) and output boundaries (before encode). Filters operate in the working space without manual conversion.

## References

- **ACES Technical Documentation:** https://acescentral.com/knowledge-base-2/
- **aces-dev GitHub (transforms):** https://github.com/AcademySoftwareFoundation/aces-dev
- **OpenColorIO ACES Config:** https://github.com/AcademySoftwareFoundation/OpenColorIO-Config-ACES
- **SMPTE ST 2065-1:** ACES Color Encoding Specification (ACES2065-1)
- **SMPTE ST 2065-3:** ACES Color Transform Language (CTL)
- **SMPTE ST 2065-4:** ACES Common LUT Format (CLF)
- **S-2014-004:** ACEScg Color Space Specification
- **S-2016-001:** ACEScct Color Space Specification
- **ISO 32000-2:2020:** PDF 2.0 (blend mode formulas, Section 11.3.5)
- **ASC CDL Specification:** https://theasc.com/asc/asc-cdl
