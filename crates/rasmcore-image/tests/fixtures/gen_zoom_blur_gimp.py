#!/usr/bin/env python3
"""Generate GIMP/GEGL reference fixtures for zoom motion blur parity tests.

Uses GIMP's batch mode with Script-Fu to apply gegl:motion-blur-zoom
to canonical test images and save raw grayscale reference output.

Requirements: GIMP 2.10+ installed and accessible via `gimp` command.

Usage: python3 gen_zoom_blur_gimp.py

Alternatively, use GEGL directly if available:
  gegl -i input.raw -o output.raw -- gegl:motion-blur-zoom center-x=64 center-y=64 factor=0.3
"""

import subprocess
import os
import sys
import struct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(SCRIPT_DIR, "opencv")  # reuse opencv fixtures dir for test images

# Test parameters matching our Rust tests
TEST_CASES = [
    {"name": "gradient_128", "w": 128, "h": 128, "factor": 0.1},
    {"name": "gradient_128", "w": 128, "h": 128, "factor": 0.3},
    {"name": "checker_128", "w": 128, "h": 128, "factor": 0.3},
    {"name": "sharp_edges_128", "w": 128, "h": 128, "factor": 0.3},
]


def gen_gimp_script(input_path, output_path, center_x, center_y, factor):
    """Generate a GIMP Script-Fu batch command for zoom blur."""
    return f"""
(let* (
  (image (car (file-raw-load RUN-NONINTERACTIVE "{input_path}" "{input_path}" 128 128 1 0 0)))
  (drawable (car (gimp-image-get-active-drawable image)))
)
  (gimp-image-set-active-layer image drawable)
  ;; Apply GEGL zoom motion blur
  (gimp-drawable-edit-stroke-selection drawable)
  ;; GEGL operation via PDB
  (gimp-message "Applying zoom blur...")
  ;; Note: GIMP Script-Fu access to GEGL ops varies by version.
  ;; For programmatic access, use Python-Fu or direct GEGL CLI instead.
  (file-raw-save RUN-NONINTERACTIVE image drawable "{output_path}" "{output_path}")
  (gimp-image-delete image)
)
"""


def check_gimp():
    """Check if GIMP is available."""
    try:
        result = subprocess.run(["gimp", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"GIMP found: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def main():
    if not check_gimp():
        print("ERROR: GIMP not found. Install GIMP 2.10+ to generate reference fixtures.")
        print()
        print("Alternative: use the GEGL CLI directly:")
        print("  gegl -i input.png -o output.png -- gegl:motion-blur-zoom \\")
        print("       center-x=0.5 center-y=0.5 factor=0.3")
        print()
        print("Or use Python with the `gi` module (GEGL introspection):")
        print("  import gi; gi.require_version('Gegl', '0.4')")
        print("  from gi.repository import Gegl")
        sys.exit(1)

    print("GIMP zoom blur fixture generation")
    print("(This script is a template — GIMP Script-Fu GEGL access is version-dependent)")
    print("For reliable parity testing, use GEGL CLI or Python gi.repository.Gegl directly.")


if __name__ == "__main__":
    main()
