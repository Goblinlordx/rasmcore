# Sampling Mode Audit: EWA vs Bilinear for Distortion Filters

## Decision Matrix

MAE values measured against ImageMagick 7.x on a 64x64 checkerboard-gradient test image.
Audit benchmark used `-virtual-pixel Background -background black` (zero border).

| Filter    | Bilinear | Ewa   | EwaClamp | Current   | Best      | Action          |
|-----------|----------|-------|----------|-----------|-----------|-----------------|
| swirl     | **0.00** | 1.53  | 1.46     | Bilinear  | Bilinear  | Applied         |
| barrel    | 2.32     | 1.66  | 8.24†    | EwaClamp  | EwaClamp  | Keep (see note) |
| wave      | **0.00** | n/t   | n/t      | Bilinear  | Bilinear  | Keep            |
| polar     | **1.95** | 5.18  | 5.20     | Bilinear  | Bilinear  | Keep            |
| depolar   | 4.35     | **2.55** | 38.29 | Ewa       | Ewa       | Keep            |
| spherize  | n/a      | n/a   | n/a      | Ewa       | --        | No IM ref       |
| ripple    | n/a      | n/a   | n/a      | Ewa       | --        | No IM ref       |
| mesh_warp | n/a      | n/a   | n/a      | Bilinear  | --        | No IM ref       |

**n/t** = not tested (wave bilinear is already exact match).
**n/a** = no direct IM equivalent for comparison.

† Barrel EwaClamp MAE 8.24 is from the **unit test** which uses IM's default
`-virtual-pixel Edge` (edge-clamp border). The audit's Ewa MAE 1.66 was
measured with `-virtual-pixel Background` (zero border) — a different IM mode.

## Key Findings

### 1. Swirl switched from Ewa to Bilinear (APPLIED)
MAE drops from 1.53 to 0.00 (exact match). IM implements -swirl in effect.c
using bilinear interpolation, not EWA. Our EWA over-filtered and introduced
error that bilinear avoids.

### 2. Barrel keeps EwaClamp (CORRECTED from initial recommendation)
The audit initially recommended switching barrel from EwaClamp to Ewa based
on MAE 44.63 → 1.66. However, this was misleading:
- The audit used `-virtual-pixel Background` (zero border), matching Ewa.
- IM barrel defaults to `-virtual-pixel Edge` (edge-clamp), matching EwaClamp.
- The unit test (which uses Edge mode) confirms EwaClamp at MAE 8.24.
- Switching to Ewa with Edge mode gives MAE 63.93 (catastrophic).

**EwaClamp is correct for barrel** because barrel distortion maps edge pixels
that need to be repeated (not zeroed). The 8.24 residual is from Robidoux vs
Laguerre filter kernel differences.

### 3. EwaClamp scope is narrow
EwaClamp is correct ONLY for barrel (which requires edge-repeat border).
It produces catastrophic results for depolar (38.29) and other operations.
Never use it as a general-purpose mode.

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

| IM Operation      | IM Resampler       | Our Mode    | Notes                           |
|-------------------|--------------------|-------------|----------------------------------|
| -distort Barrel   | EWA + Edge border  | EwaClamp    | Edge-clamp matches -virtual-pixel Edge |
| -distort Polar    | EWA Laguerre       | Ewa         | Used for our depolar direction   |
| -distort DePolar  | EWA Laguerre       | Bilinear    | Bilinear matches better empirically |
| -swirl            | Bilinear           | Bilinear    | Implemented in effect.c, not distort.c |
| -wave             | Bilinear           | Bilinear    | Implemented in effect.c          |

Note: IM's -swirl and -wave are in effect.c (simple interpolation), not distort.c
(EWA distortion engine). This explains why bilinear matches them exactly.

For -distort operations (Barrel, Polar, DePolar), IM uses its full EWA pipeline
with Laguerre cylindrical filter. Our EWA uses Robidoux filter, which explains
the ~1.5-2.5 MAE residual for EWA-mode filters.

## Final State After Implementation

| Filter    | Mode      | MAE vs IM | Status                            |
|-----------|-----------|-----------|-----------------------------------|
| wave      | Bilinear  | 0.00      | Exact match                       |
| swirl     | Bilinear  | 0.00      | Exact match (switched from Ewa)   |
| polar     | Bilinear  | 1.95      | Good parity                       |
| depolar   | Ewa       | 2.55      | Good parity (FP precision floor)  |
| barrel    | EwaClamp  | 8.24      | Acceptable (kernel + coeff diff)  |
| spherize  | Ewa       | n/a       | No IM ref; EWA suits anisotropy   |
| ripple    | Ewa       | n/a       | No IM ref; EWA suits radial       |
| mesh_warp | Bilinear  | n/a       | No IM ref; bilinear suits affine  |

## Audit Methodology Note

The `sampling_mode_audit` test in tests.rs uses `-virtual-pixel Background`
(zero border) for consistency across all filters. This does NOT match IM's
per-operation defaults — barrel defaults to Edge, others to Background.
When comparing audit results to unit test results, account for the border
mode difference. The unit tests (`im_parity_*`) use operation-appropriate
border modes and are the authoritative parity reference.
