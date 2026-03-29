//! HEVC encoder modules — direct port of x265 4.1 encoding infrastructure.
//!
//! This module provides the encoding counterpart to the decoder:
//! - `bitwrite`: RBSP bitstream writer (counterpart to `bitread`)
//! - `nal_write`: NAL unit assembly with emulation prevention (counterpart to `nal`)
//! - `params_write`: VPS/SPS/PPS serialization (counterpart to `params`)

pub mod bitwrite;
pub mod nal_write;
pub mod params_write;
