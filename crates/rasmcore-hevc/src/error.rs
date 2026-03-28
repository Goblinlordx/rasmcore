//! Error types for HEVC decoding.

use core::fmt;

/// Errors that can occur during HEVC decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HevcError {
    /// NAL unit is too short or malformed.
    InvalidNal(String),
    /// Data is truncated.
    Truncated { expected: usize, available: usize },
    /// Unsupported profile or feature.
    UnsupportedProfile(String),
    /// Parameter set error (missing or invalid VPS/SPS/PPS).
    InvalidParameterSet(String),
    /// CABAC decoding error.
    CabacError(String),
    /// General decoding error.
    DecodeFailed(String),
}

impl fmt::Display for HevcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNal(msg) => write!(f, "invalid NAL unit: {msg}"),
            Self::Truncated {
                expected,
                available,
            } => write!(f, "truncated: expected {expected} bytes, got {available}"),
            Self::UnsupportedProfile(msg) => write!(f, "unsupported profile: {msg}"),
            Self::InvalidParameterSet(msg) => write!(f, "invalid parameter set: {msg}"),
            Self::CabacError(msg) => write!(f, "CABAC error: {msg}"),
            Self::DecodeFailed(msg) => write!(f, "decode failed: {msg}"),
        }
    }
}

impl std::error::Error for HevcError {}
