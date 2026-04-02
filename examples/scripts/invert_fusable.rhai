//! name: invert_fusable
//! category: adjustment
//! strategy: fusable

// Fusable point-op invert: each channel value becomes 255 - value.
// Self-inverse: applying twice returns the original image.
// Participates in LUT fusion with adjacent point-ops.

fn lut(input, channel, params) {
    255.0 - input
}
