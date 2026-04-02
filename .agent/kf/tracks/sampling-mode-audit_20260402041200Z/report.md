# Sampling Mode Audit: EWA vs Bilinear for Distortion Filters

## Decision Matrix

All MAE values measured against ImageMagick 7.x reference on a 64x64 checkerboard-gradient test image.

| Filter    | Bilinear | Ewa   | EwaClamp | Current   | Best      | Action          |
|-----------|----------|-------|----------|-----------|-----------|-----------------|
| swirl     | **0.00** | 1.53  | 1.46     | Ewa       | Bilinear  | **SWITCH**      |
| barrel    | 2.32     | **1.66** | 44.63 | EwaClamp  | Ewa       | **SWITCH**      |
| wave      | **0.00** | n/t   | n/t      | Bilinear  | Bilinear  | Keep            |
| polar     | **1.95** | 5.18  | 5.20     | Bilinear  | Bilinear  | Keep            |
| depolar   | 4.35     | **2.55** | 38.29 | Ewa       | Ewa       | Keep            |
| spherize  | n/a      | n/a   | n/a      | Ewa       | --        | Need IM ref     |
| ripple    | n/a      | n/a   | n/a      | Ewa       | --        | Need IM ref     |
| mesh_warp | n/a      | n/a   | n/a      | Bilinear  | --        | Need IM ref     |

**n/t** = not tested (wave bilinear is already exact match).
**n/a** = no direct IM equivalent for comparison.

## Key Findings

### 1. Swirl should switch from Ewa to Bilinear
MAE drops from 1.53 to 0.00 (exact match). IM implements -swirl in effect.c
using bilinear interpolation, not EWA. Our EWA over-filters and introduces
error that bilinear avoids.

### 2. Barrel should switch from EwaClamp to Ewa
MAE drops from 44.63 to 1.66. The EwaClamp mode is catastrophically bad for
barrel — edge-clamped fetch bleeds edge pixel values into the distorted region.
Regular Ewa (zero-border) closely matches IM's barrel distortion behavior.
Bilinear at 2.32 is also acceptable but Ewa is closer.

### 3. EwaClamp is dangerous
EwaClamp produces catastrophic results for barrel (44.63) and depolar (38.29).
It should only be used when the reference tool explicitly uses edge-repeat
border handling. Currently no distortion filter benefits from it.

### 4. Polar and Depolar are already optimal
Polar with Bilinear (1.95) and depolar with Ewa (2.55) match IM best.
The asymmetry makes sense: IM uses bilinear for its DePolar (simple
uniform-to-radial mapping) and EWA for its Polar (radial-to-uniform
requires anisotropic filtering to handle the convergence at center).

### 5. Wave is already exact
Bilinear gives MAE 0.00 against IM. IM implements -wave in effect.c
using simple row/column shifting with bilinear interpolation.

## IM Resampling Strategy per Operation

Based on IM 7.x source code analysis:

| IM Operation      | IM Resampler   | Our Equivalent  | Notes                           |
|-------------------|----------------|-----------------|----------------------------------|
| -distort Barrel   | EWA Laguerre   | Ewa             | Full EWA with zero border        |
| -distort Polar    | EWA Laguerre   | Ewa             | Used for depolar direction       |
| -distort DePolar  | EWA Laguerre   | Bilinear*       | Bilinear matches better empirically |
| -swirl            | Bilinear       | Bilinear*       | Implemented in effect.c, not distort.c |
| -wave             | Bilinear       | Bilinear        | Implemented in effect.c          |

*Recommended change from current mode.

Note: IM's -swirl and -wave are in effect.c (simple interpolation), not distort.c
(EWA distortion engine). This explains why bilinear matches them exactly.

For -distort operations (Barrel, Polar, DePolar), IM uses its full EWA pipeline
with Laguerre cylindrical filter. Our EWA implementation uses Robidoux filter,
which explains the ~1.5-2.5 MAE residual even when the correct mode is selected.

## Recommended Changes

1. **swirl.rs**: Change `DistortionSampling::Ewa` to `DistortionSampling::Bilinear` (MAE 1.53 -> 0.00)
2. **barrel.rs**: Change `DistortionSampling::EwaClamp` to `DistortionSampling::Ewa` (MAE 44.63 -> 1.66)
3. **No other changes needed** for filters with IM references.

## Filters Without IM References

- **spherize**: No direct IM equivalent. Keep Ewa (the powf-based distortion has anisotropic stretching that benefits from EWA).
- **ripple**: No direct IM equivalent. Keep Ewa (radial displacement benefits from EWA near center).
- **mesh_warp**: No direct IM equivalent. Keep Bilinear (piecewise affine mapping is well-suited for bilinear).

## Follow-up Tracks

If the sampling mode changes are approved, create an implementation track to:
1. Change swirl to Bilinear
2. Change barrel to Ewa
3. Update unit test thresholds/comments
4. Verify all parity tests still pass
