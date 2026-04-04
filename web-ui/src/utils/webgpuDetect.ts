/** WebGPU feature detection for display surface. */

let _available: boolean | null = null;

/** Whether the browser supports WebGPU (navigator.gpu exists). */
export function isWebGpuAvailable(): boolean {
  if (_available !== null) return _available;
  _available = typeof navigator !== 'undefined' && 'gpu' in navigator;
  return _available;
}

/** Whether the display supports HDR (dynamic-range: high). */
export function isHdrDisplay(): boolean {
  if (typeof matchMedia === 'undefined') return false;
  return matchMedia('(dynamic-range: high)').matches;
}
