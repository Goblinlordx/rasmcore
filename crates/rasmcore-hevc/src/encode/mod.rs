//! HEVC encoder modules — direct port of x265 4.1 encoding infrastructure.
//!
//! This module provides the encoding counterpart to the decoder:
//! - `bitwrite`: RBSP bitstream writer (counterpart to `bitread`)
//! - `cabac_enc`: CABAC arithmetic encoder (counterpart to `cabac/decoder`)
//! - `encoder`: Top-level I-frame encode pipeline
//! - `nal_write`: NAL unit assembly with emulation prevention (counterpart to `nal`)
//! - `params_write`: VPS/SPS/PPS serialization (counterpart to `params`)
//! - `syntax_enc`: CABAC syntax element encoding (counterpart to `syntax`)

pub mod bitwrite;
pub mod cabac_enc;
pub mod encoder;
pub mod nal_write;
pub mod params_write;
pub mod syntax_enc;
