/**
 * Canvas color space detection and helpers for wide-gamut (Display-P3) rendering.
 *
 * Browsers with display-p3 canvas support (Chrome 121+, Firefox 128+, Safari 18+)
 * can render colors outside the sRGB gamut on wide-gamut displays. This module
 * detects support and provides consistent canvas context creation.
 */

/** Cached detection result — computed once on first call. */
let _supportsP3: boolean | null = null;

/** Detect whether the browser supports `colorSpace: 'display-p3'` on 2D canvas. */
export function supportsDisplayP3Canvas(): boolean {
  if (_supportsP3 !== null) return _supportsP3;
  try {
    const c = document.createElement('canvas');
    c.width = 1;
    c.height = 1;
    const ctx = c.getContext('2d', { colorSpace: 'display-p3' });
    _supportsP3 = ctx !== null;
  } catch {
    _supportsP3 = false;
  }
  return _supportsP3;
}

/** The color space to use for canvas contexts — 'display-p3' if supported, otherwise 'srgb'. */
export type CanvasColorSpaceName = 'display-p3' | 'srgb';

/** Returns the best available canvas color space for the current browser. */
export function preferredCanvasColorSpace(): CanvasColorSpaceName {
  return supportsDisplayP3Canvas() ? 'display-p3' : 'srgb';
}

/**
 * Get a 2D canvas context with the preferred wide-gamut color space.
 * Falls back to sRGB if display-p3 is not supported.
 */
export function getWideGamutContext(
  canvas: HTMLCanvasElement | OffscreenCanvas,
): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null {
  const colorSpace = preferredCanvasColorSpace();
  return canvas.getContext('2d', { colorSpace }) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
}
