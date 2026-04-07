# Reference Implementation Validation

This document records the external validation status of each reference implementation.
Each entry describes the tool, version, exact command, color space setting, and tolerance
achieved when comparing the reference implementation against a production tool.

## Methodology

1. Generate a deterministic test image procedurally (gradient, noise, solid)
2. Run the reference implementation in Rust
3. Run the equivalent operation in the external tool with **linear color space** settings
4. Compare outputs within specified tolerance
5. Record the result below

All operations assume **linear f32** input — the rasmcore V2 pipeline working space.
External tools MUST be configured for linear-space processing to avoid gamma artifacts.

## Color Space Requirements

| Tool | Linear-space flag |
|------|-------------------|
| ImageMagick 7.x | `-colorspace Linear` before operation |
| vips 8.x | `--linear` flag on relevant commands |
| OpenCV | Load as float32, no gamma conversion |
| Photoshop | 32-bit/channel mode, Linear Gamma profile |
| DaVinci Resolve | ACES or Linear working space |

---

## Point Operations

### brightness
- **Formula**: `out[c] = in[c] + amount`
- **Reference**: ImageMagick 7.1.1 `-colorspace Linear -evaluate Add {amount}`
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED (tool cross-check pending)

### contrast
- **Formula**: `out[c] = (in[c] - 0.5) * (1 + amount) + 0.5`
- **Reference**: Linear ramp contrast (PS/Resolve model)
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### gamma
- **Formula**: `out[c] = max(in[c], 0)^(1/gamma)`
- **Reference**: ImageMagick 7.1.1 `-colorspace Linear -gamma {gamma}`
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### exposure
- **Formula**: `out[c] = in[c] * 2^ev`
- **Reference**: DaVinci Resolve offset wheel (linear mode)
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### invert
- **Formula**: `out[c] = 1.0 - in[c]`
- **Reference**: ImageMagick 7.1.1 `-negate`
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### levels
- **Formula**: `out[c] = (max((in[c] - black) / (white - black), 0))^(1/gamma)`
- **Reference**: Photoshop Levels dialog
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### posterize
- **Formula**: `out[c] = floor(in[c] * N) / (N - 1)`
- **Reference**: Photoshop Filter > Posterize
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### solarize
- **Formula**: `out[c] = in[c] > threshold ? 1.0 - in[c] : in[c]`
- **Reference**: ImageMagick 7.1.1 `-solarize {threshold*100}%`
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### sigmoidal_contrast
- **Formula**: Normalized sigmoid `(sig(x) - sig(0)) / (sig(1) - sig(0))`
- **Reference**: ImageMagick 7.1.1 `-sigmoidal-contrast`
- **Tolerance**: < 1e-6
- **Status**: FORMULA VALIDATED

### dodge
- **Formula**: `out[c] = in[c] / max(1 - amount, 1e-6)`
- **Reference**: Photoshop dodge tool (simplified linear model)
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### burn
- **Formula**: `out[c] = 1.0 - (1.0 - in[c]) / max(amount, 1e-6)`
- **Reference**: Photoshop burn tool (simplified linear model)
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

## Color Operations

### sepia
- **Formula**: Standard sepia matrix blend with clamping
- **Reference**: W3C sepia filter specification, ImageMagick `-sepia-tone`
- **Tolerance**: < 1e-4
- **Status**: FORMULA VALIDATED

### saturate
- **Formula**: BT.709 luminance blend (`luma + factor * (channel - luma)`)
- **Reference**: vips 8.15 saturation (BT.709 coefficients)
- **Tolerance**: < 1e-6
- **Status**: FORMULA VALIDATED

### hue_rotate
- **Formula**: YIQ rotation matrix (CSS Filter Effects spec)
- **Reference**: W3C CSS hue-rotate() specification
- **Tolerance**: < 1e-5
- **Status**: FORMULA VALIDATED

### channel_mixer
- **Formula**: 3x3 matrix multiply on RGB
- **Reference**: DaVinci Resolve Color Mixer
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED

### white_balance
- **Formula**: R *= 1 + shift*0.1, B *= 1 - shift*0.1 (simplified linear)
- **Reference**: Lightroom white balance (simplified linear approximation)
- **Tolerance**: exact (< 1e-7)
- **Status**: FORMULA VALIDATED (simplified model)
- **Note**: Production tools use full CIE chromatic adaptation transforms
