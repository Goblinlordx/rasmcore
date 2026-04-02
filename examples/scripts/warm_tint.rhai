//! name: warm_tint
//! category: color
//! strategy: fusable
//! param warmth: f32 = 0.3 [0.0, 1.0]

// Fusable point-op: adds warm tint by boosting red and reducing blue.
// Engine evaluates lut() for 0-255 per channel, produces [u8; 256] LUT.
// Participates in LUT fusion with adjacent point-ops.

fn lut(input, channel, params) {
    if channel == 0 {
        // Red: boost
        clamp_value(input + params.warmth * 25.0, 0.0, 255.0)
    } else if channel == 2 {
        // Blue: reduce
        clamp_value(input - params.warmth * 15.0, 0.0, 255.0)
    } else {
        // Green: unchanged
        input
    }
}
