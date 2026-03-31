# Rust Style Guide — rasmcore

## Naming Conventions

- **Files/modules:** snake_case (`quant.rs`, `cost_engine.rs`, `pipeline_adapter.rs`)
- **Functions/methods:** snake_case (`encode_macroblock`, `pick_best_intra16`)
- **Variables:** snake_case (`mb_col`, `top_nz`, `recon_result`)
- **Types/structs/enums:** CamelCase (`QuantMatrix`, `VP8ModeScore`, `DecimateResult`)
- **Constants:** SCREAMING_SNAKE_CASE (`MAX_COST`, `NUM_BANDS`, `K_WEIGHT_TRELLIS`)
- **Type aliases:** CamelCase (`ScoreT`, `ImageInfo`)
- **Trait names:** CamelCase, descriptive verb/adjective (`Guest`, `Encoder`, `Decodable`)
- **Crate names:** kebab-case (`rasmcore-webp`, `rasmcore-codec-png`)

## Module Organization

- Organize by domain, not by file type (e.g., `src/domain/filters.rs`, not `src/utils/`)
- One primary concept per file — split large files at natural boundaries
- Co-locate unit tests in `mod tests {}` blocks at the bottom of each file
- Integration tests go in `tests/` directory at crate root
- Internal helpers stay private — only `pub` what the crate API needs

```
crates/rasmcore-{name}/
├── src/
│   ├── lib.rs           # Public API re-exports
│   ├── {domain}.rs      # Core logic
│   ├── {support}.rs     # Internal helpers
│   └── tables.rs        # Precomputed lookup tables
├── tests/
│   ├── parity.rs        # Reference comparison tests
│   └── encode_decode.rs # Round-trip tests
└── Cargo.toml
```

## Imports

Order imports with blank lines between groups:

```rust
// 1. Standard library
use std::collections::HashMap;

// 2. External crates
use thiserror::Error;

// 3. Workspace crates
use rasmcore_color::ColorSpace;

// 4. Current crate modules
use crate::quant::QuantMatrix;
use super::predict;
```

Remove unused imports immediately — they are denied by workspace lints.

## Error Handling

- Use `thiserror` for library error types with structured variants
- Avoid `anyhow` in library crates — reserve for binaries/tests only
- Return `Result<T, E>` at API boundaries; use `Option` for "not found" semantics
- Never panic in library code — return errors instead
- Never use `.unwrap()` in library code except in `const` contexts or after infallible checks with a preceding `assert!` or type guarantee

```rust
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("invalid dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("unsupported pixel format: {0:?}")]
    UnsupportedFormat(PixelFormat),
}
```

## Testing

- **TDD is strict** — write tests before implementation for new features
- Unit tests in `#[cfg(test)] mod tests {}` at the bottom of each file
- Use descriptive test names: `encode_gradient_roundtrip`, not `test1`
- Test edge cases: zero-size inputs, max values, boundary conditions
- Parity tests compare against reference implementations (cwebp, ImageMagick, ffmpeg)
- Parity tests go in `tests/parity.rs` — these are slow and may require external tools

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_zero_input_gives_zero_output() {
        let input = [0i16; 16];
        let matrix = build_matrix(64, QuantType::YAc);
        let mut output = [0i16; 16];
        quantize_block(&input, &matrix, &mut output);
        assert_eq!(output, [0i16; 16]);
    }
}
```

## Documentation

- Doc comments (`///`) on public API items only — structs, enums, public functions
- Include "Ported from" references when implementing standard algorithms
- Do not add doc comments to private functions unless the logic is non-obvious
- Do not add comments that merely restate the code

```rust
/// Trellis-optimized quantization of a 16-coefficient VP8 block.
///
/// Ported from libwebp quant_enc.c TrellisQuantizeBlock.
/// Uses Viterbi DP to find the globally optimal set of quantized levels.
pub fn trellis_quantize_block(...) -> bool {
```

## WASM Considerations

- All library crates must compile to `wasm32-wasip1` — test with `cargo component build`
- No `std::fs`, `std::net`, `std::process`, or `std::thread` in library crates
- Use `wasi:io` streams for I/O through WIT interfaces
- Memory-conscious: avoid large stack allocations (>64KB), prefer heap
- Feature-gate native-only code with `#[cfg(not(target_arch = "wasm32"))]`

## SIMD

- Use `std::arch::wasm32` intrinsics for WASM SIMD128 (primary target)
- Gate with `#[cfg(target_arch = "wasm32")]` and provide scalar fallback
- Every `unsafe` SIMD block must have a `// SAFETY:` comment

```rust
#[cfg(target_arch = "wasm32")]
{
    use std::arch::wasm32::*;
    // SAFETY: quantized and out are [i16; 16], base is 0 or 8,
    // so base..base+8 is in bounds. Stack arrays are aligned.
    unsafe {
        let q_vec = v128_load(quantized[base..].as_ptr() as *const v128);
        let step_vec = i16x8(/* ... */);
        let result = i16x8_mul(q_vec, step_vec);
        v128_store(out[base..].as_mut_ptr() as *mut v128, result);
    }
}
#[cfg(not(target_arch = "wasm32"))]
{
    for i in 0..16 {
        out[i] = quantized[i] * matrix.q[i] as i16;
    }
}
```

## Safety and `unsafe`

- Minimize `unsafe` — only for SIMD intrinsics and WIT-generated bindings
- Every `unsafe` block requires a `// SAFETY:` comment explaining:
  - What invariants the caller guarantees
  - Why the operation is sound
- Never use `unsafe` for performance without measuring — safe code is often fast enough
- Generated code (`bindings.rs`) is exempt from manual `unsafe` review

## Zero-Warning Policy

**All crates must compile with zero warnings.** This is enforced by:

- `cargo clippy --workspace -- -D warnings` in `validate.sh`
- `[workspace.lints]` in root `Cargo.toml` (see lint enforcement track)

When you encounter warnings:

| Warning | Fix | NOT this |
|---------|-----|----------|
| Unused variable `x` | Rename to `_x` | Delete it (may be intentional) |
| Unused import | Remove the import line | Add `#[allow(unused)]` |
| Dead function `f` | Prefix `_f` or add `#[allow(dead_code)]` with reason | Delete it (may be needed later) |
| Unnecessary cast `x as f64` where `x: f64` | Remove the cast | Leave it |
| Loop index: `for i in 0..v.len()` | Use `.iter().enumerate()` or `.iter()` | Leave it |
| Manual memcpy loop | Use `copy_from_slice` | Leave it |

### Platform-Specific Variables (WASM / Native)

Variables used only inside `#[cfg(target_arch = "wasm32")]` or `#[cfg(not(...))]` blocks
will trigger `unused_variables` warnings on the other target. Handle with:

```rust
// Option 1: cfg-gate the binding itself (preferred when the variable is
// only computed for one target)
#[cfg(target_arch = "wasm32")]
let simd_lut = build_simd_table();

// Option 2: use the variable on both paths (preferred when the variable
// is always needed but consumed differently)
let len = data.len();
#[cfg(target_arch = "wasm32")]
{
    simd_process(data.as_ptr(), len);
}
#[cfg(not(target_arch = "wasm32"))]
{
    scalar_process(data, len);
}

// Option 3: targeted allow (last resort — use when cfg-gating the
// binding is awkward, e.g., destructuring or multi-use)
#[allow(unused_variables)] // used only under wasm32 SIMD path
let stride = width * 4;
```

**Do NOT** blanket-`#[allow(unused)]` on modules or functions — always scope the
allow to the narrowest item.

**When adding new code**, compile-check both targets before committing:

```bash
cargo check                              # native
cargo check --target wasm32-wasip1       # WASM
```

This catches platform-conditional warnings early. CI runs both, but local
checking prevents noise for other developers.

**Critical rule:** Warning fixes must NEVER change output or behavior. They are purely cosmetic.

### Build Warning Hygiene for New Code

When adding new features, filters, encoders, or other functionality:

1. **Run `cargo clippy --workspace` before committing.** New code must not introduce warnings.
2. **Run `cargo check --target wasm32-wasip1`** if the code has any platform-conditional logic.
3. **Generated code (bindings.rs, build.rs output):** Suppress warnings at the include site with
   `#[allow(warnings)]` on the generated module — do not litter the generator with per-lint allows.
4. **New filter registrations** that add params: verify the params are used in the filter body.
   Unused params (e.g., reserved fields) should use `let _ = param;` explicitly.
5. **New encoder configs:** After adding an encoder, check that both the adapter and WIT codegen
   produce warning-free output. Run `cargo component build` to validate the full WASM build.

If CI fails on warnings, fix them **in the same PR** — do not leave warnings for a follow-up.

## Clippy Lints

Key lints enforced at workspace level:

- `needless_range_loop` — use iterators instead of index loops
- `unnecessary_cast` — remove redundant type casts
- `redundant_closure` — `.map(\|x\| foo(x))` → `.map(foo)`
- `manual_memcpy` — use `copy_from_slice` instead of byte-copy loops
- `clamp_instead_of_if` — use `.clamp(min, max)` instead of if/else chains

## Code Patterns

### Prefer iterators over index loops
```rust
// Good
for (i, pixel) in row.iter().enumerate() { ... }

// Avoid
for i in 0..row.len() { let pixel = row[i]; ... }
```

### Use struct update syntax
```rust
// Good
let config = EncodeConfig { quality: 75, ..Default::default() };

// Avoid
let mut config = EncodeConfig::default();
config.quality = 75;
```

### Const lookup tables over runtime computation
```rust
// Good — computed at compile time
const DC_TABLE: [u16; 128] = [ /* values */ ];

// Avoid — recomputed every call
fn dc_table(qp: u8) -> u16 { /* formula */ }
```
