# VP8 mb_no_coeff_skip — libvpx Reference Analysis

## Summary

The mb_no_coeff_skip implementation in our encoder was structurally correct
but the quality regression was caused by **not skipping token writes** for
skipped MBs while enabling the skip flag. With the skip flag on, the decoder
skips reading tokens for flagged MBs, causing token partition misalignment
when the encoder still writes them.

The earlier attempt DID add token skipping, but the regression persisted
because the **context zeroing** was inconsistent between the encoder's
skip path and the encoder's non-skip all-EOB path. libvpx's implementation
clarifies the exact contract.

## Key Findings from libvpx Source

### 1. Skip Decision (`mb_is_skippable`)

libvpx checks ALL blocks including Y2:
- For I16x16 (has Y2): Y blocks must have eob < 2 (DC only), Y2 must have eob == 0, UV must have eob == 0
- For B_PRED (no Y2): all 24 blocks must have eob == 0

**Critical**: Y2 MUST be all-zero for skip=true. If Y2 has any non-zero
coefficient, the MB is NOT skippable even if all Y-AC and UV are zero.

### 2. Token Writing for Skipped MBs

Two distinct paths:
- `mb_no_coeff_skip` **OFF**: `vp8_stuff_mb()` writes explicit EOB tokens for every block
- `mb_no_coeff_skip` **ON**: `vp8_fix_contexts()` zeroes context, NO tokens written at all

The token packing functions simply iterate the token list — skipped MBs
contribute zero tokens to the range.

### 3. Context Handling (`vp8_fix_contexts`)

For skipped non-B_PRED MBs:
```
memset(above_context, 0, sizeof(ENTROPY_CONTEXT_PLANES))   // ALL 9 slots zeroed
memset(left_context, 0, sizeof(ENTROPY_CONTEXT_PLANES))
```

For skipped B_PRED MBs:
```
memset(above_context, 0, sizeof(ENTROPY_CONTEXT_PLANES) - 1)  // 8 slots zeroed, Y2 preserved
memset(left_context, 0, sizeof(ENTROPY_CONTEXT_PLANES) - 1)
```

The decoder's `vp8_reset_mb_tokens_context()` does the same: zeros Y2
context only for non-4x4 modes.

### 4. prob_skip_false Computation

```
prob_skip_false = (total_mbs - skip_true_count) * 256 / total_mbs
clamped to [1, 255]
```

Note: uses `* 256` not `* 255`. This is a subtle difference from our implementation.

### 5. Non-Skipped All-Zero Y2

When an MB is NOT skipped but Y2 is all-zero, libvpx STILL writes
`DCT_EOB_TOKEN` for the Y2 block and sets Y2 context to 0. The decoder
reads this EOB token and also sets context to 0. This produces the SAME
context state as skip — so for all-zero MBs, skip and non-skip produce
identical context states.

## Root Cause of Our Quality Regression

Our implementation had the **context zeroing correct** (matching libvpx).
The quality regression was NOT from context mismatch but from a **stale
working tree** issue during development — the skip changes were being
tested against a branch that lacked the segmentation code, causing the
encoder to use a single QP instead of per-segment QPs. This produced
different coefficient values and different skip decisions.

When tested on the actual main branch with segmentation, the skip
implementation should work correctly because:

1. Our `all_zero` check tests Y2 + Y + UV (matching libvpx's `mb_is_skippable`)
2. Our context zeroing matches `vp8_fix_contexts`
3. Our token skipping (not writing tokens) matches the `mb_no_coeff_skip=ON` path
4. Our prob computation is close (255 vs 256 scaling — minor)

## Implementation Spec for webp-skip-flag Track

### Changes Required (3 locations in bitstream.rs)

**1. First partition header** (encode_first_partition):
```rust
// Compute skip statistics
let skip_count = mb_infos.iter().filter(|m| m.skip).count();
let total = mb_infos.len().max(1);
let enable_skip = skip_count > 0;

if enable_skip {
    bw.put_literal(1, 1);  // mb_no_coeff_skip = true
    // Note: libvpx uses * 256, not * 255
    let prob = ((total - skip_count) * 256 / total).clamp(1, 255) as u8;
    bw.put_literal(8, prob as u32);
} else {
    bw.put_literal(1, 0);
}
```

**2. Per-MB header** (in MB loop, AFTER segment ID, BEFORE mode):
```rust
if enable_skip {
    bw.put_bit(prob_skip, mb.skip);
}
```

**3. Token encoding** (encode_macroblock):
Move `all_zero` computation BEFORE token section (after coefficient computation).
When `all_zero = true`:
```rust
if all_zero {
    // Match vp8_fix_contexts: zero all contexts
    if !use_bpred {
        top_ctx[0] = 0;  // Y2 context
        left_ctx[0] = 0;
    }
    // Y, U, V contexts
    for i in 1..9 {
        top_ctx[i] = 0;
        left_ctx[i] = 0;
    }
    // DO NOT write any tokens to token_bw
}
```

### Critical Validation

- Must ensure the working tree has ALL current main features (segmentation, B_PRED, etc.)
- `git checkout main -- crates/rasmcore-webp/src/` before applying changes
- Test with `cargo test -p rasmcore-webp` (all 3 test suites)
- Verify gradient PSNR does not regress from baseline
