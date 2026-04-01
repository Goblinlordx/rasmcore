# File Organization

## One filter per file

Every filter MUST be in its own file under `src/domain/filters/<category>/<name>.rs`.

Each filter file contains:
- The `#[register_filter/mapper/compositor/generator]` annotation + pub fn
- The `ConfigParams` struct (if any) + its impl blocks
- Private helper functions used ONLY by this filter
- `use crate::domain::filters::common::*;` for shared types and helpers

## Shared code

Shared helpers, enums, constants, and structs live in `filters/common.rs`.
Individual filter files MUST NOT call other filter functions directly — use
shared helpers from `common.rs` or domain modules (`point_ops`, `color_grading`, etc.).

## When adding a new filter

1. Create `filters/<category>/<name>.rs`
2. Add `mod <name>; pub use <name>::*;` to `filters/<category>/mod.rs`
3. Add ConfigParams struct + registration annotation + function
4. `use crate::domain::filters::common::*;` for shared imports
5. Verify: `cargo build && cargo component build`

## Agent concurrency for bulk operations

When a track involves moving/splitting many files, use subagents with
`isolation: "worktree"` to handle batches in parallel. Parent agent reviews
each subagent's diff before merging.
