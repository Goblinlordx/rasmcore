# Standard WIT Analysis

## Research Date: 2026-03-27

---

## WIT Language Reference Summary

### Primitive Types

`bool`, `u8`-`u64`, `s8`-`s64`, `f32`, `f64`, `char`, `string`

### Compound Types

| Type | Syntax | Purpose |
|------|--------|---------|
| Record | `record name { field: type }` | Named struct |
| Variant | `variant name { case1, case2(type) }` | Tagged union |
| Enum | `enum name { a, b, c }` | Variant without payload |
| Flags | `flags name { flag1, flag2 }` | Bitfield set |
| List | `list<T>` | Dynamic array |
| Option | `option<T>` | Nullable |
| Result | `result<T, E>` | Error handling |
| Tuple | `tuple<T1, T2>` | Fixed-length indexed |

### Resources

```wit
resource blob {
    constructor(init: list<u8>);
    write: func(bytes: list<u8>);
    read: func(n: u32) -> list<u8>;
    merge: static func(lhs: blob, rhs: blob) -> blob;
}
```

- **Owned**: Passed by value, caller transfers ownership
- **Borrowed**: `borrow<T>`, temporary loan for duration of call
- Resources have constructors, methods, and static functions

### Interfaces, Worlds, Packages

```wit
package razm:core@0.1.0;

interface types {
    type buffer = list<u8>;
    record point { x: u32, y: u32 }
}

interface image-processor {
    use types.{buffer};
    resize: func(input: buffer, width: u32, height: u32) -> buffer;
}

world image-tool {
    import wasi:io/streams@0.2.8;
    export image-processor;
}
```

- **Package**: Namespace + name + optional semver (`razm:core@0.1.0`)
- **Interface**: Collection of types and functions
- **World**: Component contract — what it imports/exports
- **Use**: Import types across interfaces (`use types.{buffer}`)
- **Include**: Compose worlds (`include base-world;`)

---

## wasi:io Analysis

### Package: `wasi:io@0.2.8`

Three interfaces: `poll`, `streams`, `error`

#### poll Interface

```wit
resource pollable {
    ready: func() -> bool;
    block: func();
}
poll: func(in: list<borrow<pollable>>) -> list<u32>;
```

- Multiplexed I/O waiting across handles
- Timeout via adding `wasi:clocks` pollable to list
- Traps on empty list

#### streams Interface

**input-stream resource:**
- `read(len: u64) → result<list<u8>, stream-error>` — non-blocking
- `blocking-read(len: u64) → result<list<u8>, stream-error>` — blocks until ≥1 byte
- `skip(len: u64)` / `blocking-skip(len: u64)` — advance without reading
- `subscribe() → pollable` — for async waiting

**output-stream resource:**
- `check-write() → result<u64, stream-error>` — capacity query
- `write(contents: list<u8>)` — non-blocking (must check-write first, traps if over limit)
- `blocking-write-and-flush(contents: list<u8>)` — up to 4096 bytes
- `flush()` / `blocking-flush()` — flush buffered output
- `splice(src: borrow<input-stream>, len: u64)` — stream-to-stream transfer
- `blocking-splice(...)` — blocking version

**stream-error variant:**
- `last-operation-failed(error)` — operation failed, stream closed
- `closed` — stream is shut down

### Design Patterns Observed

1. **Non-blocking first**: Every operation has a non-blocking form; blocking forms are convenience wrappers
2. **Check-then-write**: `check-write` → `write` pattern prevents overruns (traps if violated)
3. **Pollable subscription**: Resources provide `subscribe()` for async integration
4. **Splice for zero-copy**: Direct stream-to-stream transfer without intermediate buffers
5. **Error closes stream**: After `last-operation-failed`, stream is permanently closed

---

## wasi:filesystem Analysis

### Package: `wasi:filesystem@0.2.8`

#### Key Types

- `type filesize = u64`
- `enum descriptor-type { unknown, block-device, character-device, directory, fifo, symbolic-link, regular-file, socket }`
- `flags descriptor-flags { read, write, file-integrity-sync, data-integrity-sync, requested-write-sync, mutate-directory }`
- `record descriptor-stat { type, link-count, size, timestamps... }`

#### descriptor Resource

**Stream-based file I/O:**
```wit
read-via-stream: func(offset: filesize) -> result<input-stream, error-code>;
write-via-stream: func(offset: filesize) -> result<output-stream, error-code>;
append-via-stream: func() -> result<output-stream, error-code>;
```

**Path-based operations:**
```wit
open-at: func(path-flags, path: string, open-flags, flags) -> result<descriptor, error-code>;
stat-at: func(path-flags, path: string) -> result<descriptor-stat, error-code>;
```

### Design Patterns Observed

1. **Capability-based security**: All paths are relative to a pre-opened descriptor. No absolute paths. `..` escape fails with `not-permitted`
2. **Streams for I/O**: File content accessed via `input-stream`/`output-stream`, not raw read/write syscalls
3. **Resources as handles**: `descriptor` is a resource — ownership/borrowing applies
4. **POSIX-aligned errors**: `error-code` enum maps to familiar POSIX errors
5. **Identity without inodes**: `is-same-object()` checks identity without exposing internal IDs

---

## wasi:http Analysis (from ecosystem research)

### wasi:http@0.2.4 → wasi:http@0.3.0 Evolution

| Aspect | 0.2 | 0.3 |
|--------|-----|-----|
| Resource types | 11 | 5 |
| Async model | Pollable-based | Native `stream<T>` and `future<T>` |
| Interface complexity | High | Significantly reduced |

### Design Patterns Observed

1. **Request/Response resources**: HTTP modeled as resource types with methods
2. **Incoming vs outgoing**: Separate handler interfaces for different roles
3. **Body as stream**: Request/response bodies are streams, not buffers
4. **Simplification trend**: 0.3 drastically simplifies by leveraging native async

---

## WASI 0.3 Async Types — Key for razm/core

### New Built-in Types

```wit
// WASI 0.3 native types (built into Component Model)
stream<T>    // Typed streaming data
future<T>    // Single async value
```

### Impact on Interface Design

In 0.2, async requires manual pollable patterns:
```wit
// 0.2 pattern — verbose
resource processor {
    start: func(input: input-stream) -> pollable;
    get-result: func() -> option<output-stream>;
}
```

In 0.3, async is native:
```wit
// 0.3 pattern — clean
process: async func(input: stream<u8>) -> stream<u8>;
```

### Structured Streams

0.3 allows typed streams beyond `stream<u8>`:
```wit
stream<log-entry>     // Stream of structured records
stream<video-frame>   // Stream of frame data
stream<arrow-batch>   // Stream of data batches
```

**This is transformative for razm/core** — we can define typed streaming interfaces for frames, samples, and data records rather than raw byte streams.

---

## Key Takeaways for razm/core Interface Design

1. **Use `list<u8>` as buffer type** — the standard pattern for byte data (equivalent to `type buffer = list<u8>`)
2. **Model long-running operations as resources** — resources have lifecycle (create → use → drop)
3. **Design for 0.3 migration** — use stream-based APIs in 0.2 that map cleanly to `stream<T>` in 0.3
4. **Follow capability-based security** — components receive handles to resources, not paths/names
5. **Non-blocking by default** — provide blocking variants as convenience, not the primary API
6. **Use `result<T, E>` everywhere** — explicit error handling, no exceptions
7. **Share types via dedicated interfaces** — define `razm:core/types` for shared types, `use` in domain interfaces
8. **Version with semver** — `razm:core@0.1.0`, `razm:image@0.1.0`, etc.

---

## Sources

- [wasi:io streams.wit](https://github.com/WebAssembly/wasi-io/blob/main/wit/streams.wit)
- [wasi:io poll.wit](https://github.com/WebAssembly/wasi-io/blob/main/wit/poll.wit)
- [wasi:filesystem types.wit](https://github.com/WebAssembly/wasi-filesystem/blob/main/wit/types.wit)
- [WIT Reference — Component Model](https://component-model.bytecodealliance.org/design/wit.html)
- [WASI 0.3 Native Async](https://byteiota.com/wasi-0-3-native-async-webassembly-gets-concurrent-i-o/)
- [Looking Ahead to WASIp3 — Fermyon](https://www.fermyon.com/blog/looking-ahead-to-wasip3)
- [WASI Standards Evolution 0.2 to 0.3](https://wasmruntime.com/en/blog/wasi-standards-evolution-0.2-to-0.3)
