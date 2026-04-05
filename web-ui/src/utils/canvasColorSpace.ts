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

/**
 * Get a 2D canvas context with the best available color space.
 *
 * - If the browser supports display-p3, uses `{ colorSpace: 'display-p3' }`.
 * - Otherwise, calls `getContext('2d')` with NO options — passing
 *   `{ colorSpace: 'srgb' }` can return null in browsers that don't
 *   recognize the colorSpace option at all.
 * - If the preferred call returns null (e.g., context already exists with
 *   different attributes), falls back to plain `getContext('2d')`.
 */
export function getWideGamutContext(
  canvas: HTMLCanvasElement | OffscreenCanvas,
): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null {
  if (supportsDisplayP3Canvas()) {
    const ctx = canvas.getContext('2d', { colorSpace: 'display-p3' });
    if (ctx) return ctx as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  }
  // Fallback: plain getContext with no options — maximum browser compatibility
  return canvas.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
}
