# WIT Interface Definitions — Draft

## Date: 2026-03-27
## Project: rasmcore

---

## Package Structure

```
rasmcore:core@0.1.0     — Shared types, errors
rasmcore:image@0.1.0    — Image processing interfaces
rasmcore:video@0.1.0    — Video processing interfaces
rasmcore:audio@0.1.0    — Audio processing interfaces
rasmcore:data@0.1.0     — Data processing interfaces
rasmcore:codec@0.1.0    — Codec plugin interfaces (encoder/decoder contracts)
```

---

## rasmcore:core@0.1.0 — Shared Types

```wit
package rasmcore:core@0.1.0;

/// Shared types used across all rasmcore modules
interface types {
    /// Raw byte buffer — universal data container
    type buffer = list<u8>;

    /// Pixel formats for image/video frame data
    enum pixel-format {
        rgb8,
        rgba8,
        bgr8,
        bgra8,
        gray8,
        gray16,
        yuv420p,
        yuv422p,
        yuv444p,
        nv12,
    }

    /// Color space identifiers
    enum color-space {
        srgb,
        linear-srgb,
        display-p3,
        bt709,
        bt2020,
    }

    /// Audio sample formats
    enum sample-format {
        f32-le,
        s16-le,
        s32-le,
        u8,
    }

    /// Image dimensions and format metadata
    record image-info {
        width: u32,
        height: u32,
        format: pixel-format,
        color-space: color-space,
    }

    /// Video frame metadata
    record frame-info {
        width: u32,
        height: u32,
        format: pixel-format,
        timestamp-us: s64,
        duration-us: s64,
        keyframe: bool,
    }

    /// Audio buffer metadata
    record audio-info {
        sample-rate: u32,
        channels: u16,
        format: sample-format,
        frame-count: u32,
    }

    /// Codec identification
    record codec-info {
        id: string,
        name: string,
        mime-type: string,
        is-patent-encumbered: bool,
    }
}

/// Error types shared across modules
interface errors {
    variant rasmcore-error {
        invalid-input(string),
        unsupported-format(string),
        codec-error(string),
        io-error(string),
        out-of-memory,
        not-implemented,
        invalid-state(string),
    }
}
```

---

## rasmcore:image@0.1.0 — Image Processing

```wit
package rasmcore:image@0.1.0;

/// Image format detection and decoding
interface decoder {
    use rasmcore:core/types.{buffer, image-info};
    use rasmcore:core/errors.{rasmcore-error};

    /// Detect image format from header bytes
    detect-format: func(header: buffer) -> option<string>;

    /// Decode an image from a buffer
    /// Returns pixel data and metadata
    decode: func(data: buffer) -> result<decoded-image, rasmcore-error>;

    /// Decode with specific output format request
    decode-as: func(data: buffer, target-format: pixel-format) -> result<decoded-image, rasmcore-error>;

    record decoded-image {
        pixels: buffer,
        info: image-info,
    }

    /// List supported decode formats (e.g., "png", "jpeg", "webp")
    supported-formats: func() -> list<string>;

    use rasmcore:core/types.{pixel-format};
}

/// Image encoding
interface encoder {
    use rasmcore:core/types.{buffer, image-info};
    use rasmcore:core/errors.{rasmcore-error};

    /// Encode pixel data to a specific format
    encode: func(pixels: buffer, info: image-info, format: string, quality: option<u8>) -> result<buffer, rasmcore-error>;

    /// List supported encode formats
    supported-formats: func() -> list<string>;
}

/// Image manipulation operations
interface transform {
    use rasmcore:core/types.{buffer, image-info};
    use rasmcore:core/errors.{rasmcore-error};

    /// Resize an image
    resize: func(pixels: buffer, info: image-info, width: u32, height: u32, filter: resize-filter) -> result<tuple<buffer, image-info>, rasmcore-error>;

    enum resize-filter {
        nearest,
        bilinear,
        bicubic,
        lanczos3,
    }

    /// Crop an image
    crop: func(pixels: buffer, info: image-info, x: u32, y: u32, width: u32, height: u32) -> result<tuple<buffer, image-info>, rasmcore-error>;

    /// Rotate an image
    rotate: func(pixels: buffer, info: image-info, degrees: rotation) -> result<tuple<buffer, image-info>, rasmcore-error>;

    enum rotation {
        r90,
        r180,
        r270,
    }

    /// Flip an image
    flip: func(pixels: buffer, info: image-info, direction: flip-direction) -> result<tuple<buffer, image-info>, rasmcore-error>;

    enum flip-direction {
        horizontal,
        vertical,
    }

    /// Convert pixel format
    convert-format: func(pixels: buffer, info: image-info, target: pixel-format) -> result<tuple<buffer, image-info>, rasmcore-error>;

    use rasmcore:core/types.{pixel-format};
}

/// Image filters and effects
interface filters {
    use rasmcore:core/types.{buffer, image-info};
    use rasmcore:core/errors.{rasmcore-error};

    /// Apply gaussian blur
    blur: func(pixels: buffer, info: image-info, radius: f32) -> result<buffer, rasmcore-error>;

    /// Apply sharpening
    sharpen: func(pixels: buffer, info: image-info, amount: f32) -> result<buffer, rasmcore-error>;

    /// Adjust brightness (-1.0 to 1.0)
    brightness: func(pixels: buffer, info: image-info, amount: f32) -> result<buffer, rasmcore-error>;

    /// Adjust contrast (-1.0 to 1.0)
    contrast: func(pixels: buffer, info: image-info, amount: f32) -> result<buffer, rasmcore-error>;

    /// Convert to grayscale
    grayscale: func(pixels: buffer, info: image-info) -> result<tuple<buffer, image-info>, rasmcore-error>;
}

/// World for a complete image processing component
world image-processor {
    import wasi:io/streams@0.2.8;
    export decoder;
    export encoder;
    export transform;
    export filters;
}
```

---

## rasmcore:codec@0.1.0 — Codec Plugin Interfaces

```wit
package rasmcore:codec@0.1.0;

/// Video/audio encoder plugin interface
interface encoder {
    use rasmcore:core/types.{buffer, frame-info, codec-info};
    use rasmcore:core/errors.{rasmcore-error};

    /// Stateful encoder resource
    resource encoder-instance {
        constructor(config: encoder-config);

        /// Push a frame for encoding
        /// Returns zero or more encoded packets
        push-frame: func(data: buffer, info: frame-info) -> result<list<encoded-packet>, rasmcore-error>;

        /// Flush remaining frames
        flush: func() -> result<list<encoded-packet>, rasmcore-error>;

        /// Get current encoder stats
        stats: func() -> encoder-stats;
    }

    record encoder-config {
        codec-id: string,
        width: u32,
        height: u32,
        framerate-num: u32,
        framerate-den: u32,
        bitrate-bps: u64,
        /// Codec-specific key-value parameters
        params: list<tuple<string, string>>,
    }

    record encoded-packet {
        data: buffer,
        timestamp-us: s64,
        duration-us: s64,
        keyframe: bool,
    }

    record encoder-stats {
        frames-in: u64,
        packets-out: u64,
        bytes-out: u64,
    }

    /// Discover what this plugin can encode
    supported-codecs: func() -> list<codec-info>;
}

/// Video/audio decoder plugin interface
interface decoder {
    use rasmcore:core/types.{buffer, frame-info, codec-info};
    use rasmcore:core/errors.{rasmcore-error};

    /// Stateful decoder resource
    resource decoder-instance {
        constructor(codec-id: string);

        /// Push an encoded packet for decoding
        /// Returns zero or more decoded frames
        push-packet: func(data: buffer) -> result<list<decoded-frame>, rasmcore-error>;

        /// Flush remaining frames
        flush: func() -> result<list<decoded-frame>, rasmcore-error>;
    }

    record decoded-frame {
        data: buffer,
        info: frame-info,
    }

    /// Discover what this plugin can decode
    supported-codecs: func() -> list<codec-info>;
}

/// World for a codec plugin component
world codec-plugin {
    export encoder;
    export decoder;
}

/// World for encoder-only plugin
world encoder-plugin {
    export encoder;
}

/// World for decoder-only plugin
world decoder-plugin {
    export decoder;
}
```

---

## rasmcore:video@0.1.0 — Video Processing

```wit
package rasmcore:video@0.1.0;

/// Container demuxing — extract streams from container formats
interface demuxer {
    use rasmcore:core/types.{buffer, codec-info, frame-info, audio-info};
    use rasmcore:core/errors.{rasmcore-error};

    /// Open a container and inspect its streams
    resource container {
        /// Create from a complete buffer (small files)
        constructor(data: buffer);

        /// Get number of streams
        stream-count: func() -> u32;

        /// Get stream info
        stream-info: func(index: u32) -> result<stream-metadata, rasmcore-error>;

        /// Read next packet from any stream
        read-packet: func() -> result<option<demuxed-packet>, rasmcore-error>;

        /// Seek to timestamp (microseconds)
        seek: func(timestamp-us: s64) -> result<_, rasmcore-error>;
    }

    record stream-metadata {
        index: u32,
        stream-type: stream-type,
        codec: codec-info,
        duration-us: option<s64>,
    }

    enum stream-type {
        video,
        audio,
        subtitle,
        data,
    }

    record demuxed-packet {
        stream-index: u32,
        data: buffer,
        timestamp-us: s64,
        duration-us: s64,
        keyframe: bool,
    }

    /// List supported container formats
    supported-formats: func() -> list<string>;
}

/// Container muxing — write streams to container formats
interface muxer {
    use rasmcore:core/types.{buffer, codec-info};
    use rasmcore:core/errors.{rasmcore-error};

    resource mux-session {
        constructor(format: string);

        /// Add a stream to the output
        add-stream: func(codec: codec-info) -> result<u32, rasmcore-error>;

        /// Write a packet to a stream
        write-packet: func(stream-index: u32, data: buffer, timestamp-us: s64, duration-us: s64, keyframe: bool) -> result<_, rasmcore-error>;

        /// Finalize and get the output
        finalize: func() -> result<buffer, rasmcore-error>;
    }

    /// List supported output container formats
    supported-formats: func() -> list<string>;
}

/// World for a complete video processing component
world video-processor {
    import wasi:io/streams@0.2.8;
    import rasmcore:codec/encoder;
    import rasmcore:codec/decoder;
    export demuxer;
    export muxer;
}
```

---

## rasmcore:audio@0.1.0 — Audio Processing

```wit
package rasmcore:audio@0.1.0;

/// Audio manipulation operations
interface transform {
    use rasmcore:core/types.{buffer, audio-info};
    use rasmcore:core/errors.{rasmcore-error};

    /// Resample audio to target sample rate
    resample: func(data: buffer, info: audio-info, target-rate: u32) -> result<tuple<buffer, audio-info>, rasmcore-error>;

    /// Convert sample format
    convert-format: func(data: buffer, info: audio-info, target: sample-format) -> result<tuple<buffer, audio-info>, rasmcore-error>;

    /// Mix channels (e.g., stereo to mono)
    mix-channels: func(data: buffer, info: audio-info, target-channels: u16) -> result<tuple<buffer, audio-info>, rasmcore-error>;

    /// Adjust volume (gain in dB)
    gain: func(data: buffer, info: audio-info, db: f32) -> result<buffer, rasmcore-error>;

    /// Trim audio by sample range
    trim: func(data: buffer, info: audio-info, start-sample: u32, end-sample: u32) -> result<tuple<buffer, audio-info>, rasmcore-error>;

    use rasmcore:core/types.{sample-format};
}
```

---

## rasmcore:data@0.1.0 — Data Processing

```wit
package rasmcore:data@0.1.0;

/// Format conversion between data serialization formats
interface converter {
    use rasmcore:core/types.{buffer};
    use rasmcore:core/errors.{rasmcore-error};

    /// Convert between data formats
    convert: func(input: buffer, from-format: string, to-format: string) -> result<buffer, rasmcore-error>;

    /// List supported data formats (e.g., "json", "csv", "msgpack", "parquet")
    supported-formats: func() -> list<string>;

    /// Detect format from content
    detect-format: func(data: buffer) -> option<string>;
}

/// Tabular data operations
interface table {
    use rasmcore:core/types.{buffer};
    use rasmcore:core/errors.{rasmcore-error};

    /// Load tabular data from a supported format
    resource dataframe {
        constructor(data: buffer, format: string);

        /// Get row and column counts
        shape: func() -> tuple<u64, u32>;

        /// Get column names
        columns: func() -> list<string>;

        /// Select columns by name
        select: func(columns: list<string>) -> result<dataframe, rasmcore-error>;

        /// Filter rows by expression
        filter: func(column: string, op: filter-op, value: string) -> result<dataframe, rasmcore-error>;

        /// Sort by column
        sort: func(column: string, descending: bool) -> result<dataframe, rasmcore-error>;

        /// Export to format
        to-format: func(format: string) -> result<buffer, rasmcore-error>;

        /// Get first N rows
        head: func(n: u64) -> result<dataframe, rasmcore-error>;
    }

    enum filter-op {
        eq,
        neq,
        gt,
        gte,
        lt,
        lte,
        contains,
    }
}

world data-processor {
    import wasi:io/streams@0.2.8;
    export converter;
    export table;
}
```

---

## Interface Design Principles Applied

1. **Coarse-grained APIs** — Functions take/return whole buffers, not individual pixels/samples
2. **Resources for stateful ops** — Encoders, decoders, containers, dataframes have lifecycle
3. **`result<T, rasmcore-error>` everywhere** — Explicit error handling
4. **`supported-*()` discovery functions** — Plugins declare capabilities
5. **Stream-ready** — Uses `wasi:io/streams` imports; migrates cleanly to 0.3 `stream<T>`
6. **Minimal shared types** — Only `rasmcore:core/types` crosses domain boundaries
7. **Separate encode/decode** — Plugins can implement one or both
8. **Format as string** — Codec/format IDs are strings for extensibility
