#!/usr/bin/env python3
"""Generate a minimal synthetic DNG file for reference parity testing.

Creates a 64×64 uncompressed DNG with RGGB Bayer CFA, realistic color
matrix, and known sensor data (gradient pattern). This file must be
decodable by dcraw, rawpy, AND our rasmcore-raw decoder.

DNG spec: https://helpx.adobe.com/photoshop/digital-negative.html
Based on TIFF/EP (ISO 12234-2) with DNG extensions.
"""

import struct
import sys
import os

# --- DNG Parameters ---
WIDTH = 64
HEIGHT = 64
BPS = 16  # bits per sample
CFA = [0, 1, 1, 2]  # RGGB: Red=0, Green=1, Blue=2

# Realistic color matrix from a Nikon D7000 (D65 illuminant)
# ColorMatrix1: XYZ → camera (SRATIONAL, stored as num/den pairs)
COLOR_MATRIX = [
    (8198, 10000), (-2239, 10000), (-724, 10000),
    (-4871, 10000), (12389, 10000), (2798, 10000),
    (-1043, 10000), (2050, 10000), (7181, 10000),
]

# AsShotNeutral: camera neutral for daylight (RATIONAL)
AS_SHOT_NEUTRAL = [
    (4587, 10000),  # R neutral
    (10000, 10000), # G neutral
    (7012, 10000),  # B neutral
]

# CalibrationIlluminant1 = 21 (D65)
CALIBRATION_ILLUMINANT = 21

# --- Generate raw sensor data: smooth gradient ---
def generate_gradient_raw(w, h):
    """Generate a gradient Bayer mosaic: brightness increases left→right."""
    raw = []
    for row in range(h):
        for col in range(w):
            # Base value: gradient across columns
            base = int((col / (w - 1)) * 60000) + 256
            # Add slight row variation
            base += int((row / (h - 1)) * 2000)
            base = min(base, 65535)
            raw.append(base)
    return raw


def write_u16(f, val):
    f.write(struct.pack('<H', val))

def write_u32(f, val):
    f.write(struct.pack('<I', val))

def write_i32(f, val):
    f.write(struct.pack('<i', val))


def build_dng(output_path):
    """Build a minimal valid DNG file."""
    raw_pixels = generate_gradient_raw(WIDTH, HEIGHT)

    # Raw data as bytes (16-bit LE)
    raw_bytes = b''.join(struct.pack('<H', v) for v in raw_pixels)

    # --- Layout plan ---
    # Offset 0: TIFF header (8 bytes)
    # Offset 8: Raw pixel data (WIDTH * HEIGHT * 2 bytes)
    # After raw: Extended tag values (ColorMatrix, AsShotNeutral, etc.)
    # After values: IFD0

    raw_data_offset = 8
    raw_data_size = len(raw_bytes)
    ext_start = 8 + raw_data_size

    # Pre-serialize extended values and track their offsets
    ext_values = bytearray()

    # ColorMatrix1: 9 × SRATIONAL (9 × 8 = 72 bytes)
    cm_offset = ext_start + len(ext_values)
    for num, den in COLOR_MATRIX:
        ext_values += struct.pack('<i', num)
        ext_values += struct.pack('<i', den)

    # AsShotNeutral: 3 × RATIONAL (3 × 8 = 24 bytes)
    asn_offset = ext_start + len(ext_values)
    for num, den in AS_SHOT_NEUTRAL:
        ext_values += struct.pack('<I', num)
        ext_values += struct.pack('<I', den)

    # UniqueCameraModel: ASCII string (include null terminator)
    model_str = b"Synthetic DNG Test\x00"
    model_offset = ext_start + len(ext_values)
    ext_values += model_str

    # DNGBackwardVersion: [1, 1, 0, 0] (BYTE × 4, fits inline)
    # Software: ASCII
    software_str = b"rasmcore-raw test fixture\x00"
    software_offset = ext_start + len(ext_values)
    ext_values += software_str

    # --- Build IFD entries ---
    # TIFF tag types: BYTE=1, ASCII=2, SHORT=3, LONG=4, RATIONAL=5,
    #                 SBYTE=6, UNDEFINED=7, SSHORT=8, SLONG=9, SRATIONAL=10
    ifd_entries = []

    def add_tag(tag, typ, count, value):
        """Add IFD entry. value is the 4-byte value/offset field."""
        ifd_entries.append((tag, typ, count, value))

    # Standard TIFF tags
    add_tag(254, 4, 1, 0)                     # NewSubFileType = 0 (full-res)
    add_tag(256, 4, 1, WIDTH)                  # ImageWidth
    add_tag(257, 4, 1, HEIGHT)                 # ImageLength
    add_tag(258, 3, 1, BPS)                    # BitsPerSample = 16
    add_tag(259, 3, 1, 1)                      # Compression = none
    add_tag(262, 3, 1, 32803)                  # PhotometricInterpretation = CFA
    add_tag(273, 4, 1, raw_data_offset)        # StripOffsets
    add_tag(274, 3, 1, 1)                      # Orientation = top-left
    add_tag(277, 3, 1, 1)                      # SamplesPerPixel = 1
    add_tag(278, 4, 1, HEIGHT)                 # RowsPerStrip = HEIGHT
    add_tag(279, 4, 1, raw_data_size)          # StripByteCounts

    # Software tag
    add_tag(305, 2, len(software_str), software_offset)

    # CFA tags
    # CFARepeatPatternDim (33421): SHORT × 2 = [2, 2]
    cfa_dim_val = struct.pack('<HH', 2, 2)
    add_tag(33421, 3, 2, struct.unpack('<I', cfa_dim_val)[0])

    # CFAPattern (33422): BYTE × 4 = [0, 1, 1, 2]
    cfa_val = struct.pack('BBBB', *CFA)
    add_tag(33422, 1, 4, struct.unpack('<I', cfa_val)[0])

    # DNG tags
    # DNGVersion (50706): BYTE × 4 = [1, 4, 0, 0]
    dng_ver = struct.pack('BBBB', 1, 4, 0, 0)
    add_tag(50706, 1, 4, struct.unpack('<I', dng_ver)[0])

    # DNGBackwardVersion (50707): BYTE × 4 = [1, 1, 0, 0]
    dng_bver = struct.pack('BBBB', 1, 1, 0, 0)
    add_tag(50707, 1, 4, struct.unpack('<I', dng_bver)[0])

    # UniqueCameraModel (50708): ASCII
    add_tag(50708, 2, len(model_str), model_offset)

    # BlackLevel (50714): RATIONAL × 1 (= 256.0)
    bl_offset = ext_start + len(ext_values)
    ext_values += struct.pack('<II', 256, 1)  # 256/1
    add_tag(50714, 5, 1, bl_offset)

    # WhiteLevel (50717): LONG × 1
    add_tag(50717, 4, 1, 65535)

    # ColorMatrix1 (50721): SRATIONAL × 9
    add_tag(50721, 10, 9, cm_offset)

    # AsShotNeutral (50727): RATIONAL × 3
    add_tag(50727, 5, 3, asn_offset)

    # CalibrationIlluminant1 (50778): SHORT × 1
    add_tag(50778, 3, 1, CALIBRATION_ILLUMINANT)

    # Sort by tag number (TIFF requirement)
    ifd_entries.sort(key=lambda x: x[0])

    # --- Compute IFD offset ---
    ifd_offset = ext_start + len(ext_values)

    # --- Write the file ---
    with open(output_path, 'wb') as f:
        # TIFF header
        f.write(b'II')                            # Little-endian
        write_u16(f, 42)                          # TIFF magic
        write_u32(f, ifd_offset)                  # Offset to IFD0

        # Raw pixel data
        f.write(raw_bytes)

        # Extended tag values
        f.write(bytes(ext_values))

        # IFD0
        write_u16(f, len(ifd_entries))            # Number of entries
        for tag, typ, count, value in ifd_entries:
            write_u16(f, tag)
            write_u16(f, typ)
            write_u32(f, count)
            write_u32(f, value)
        write_u32(f, 0)                           # Next IFD offset = 0

    print(f"Written: {output_path} ({os.path.getsize(output_path)} bytes)")
    print(f"  Dimensions: {WIDTH}×{HEIGHT}, {BPS}-bit")
    print(f"  CFA: RGGB, Compression: none")
    print(f"  Raw data: {raw_data_size} bytes at offset {raw_data_offset}")
    print(f"  IFD at offset {ifd_offset}, {len(ifd_entries)} entries")


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'tests/fixtures/generated/inputs/gradient_64x64.dng'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_dng(out)
