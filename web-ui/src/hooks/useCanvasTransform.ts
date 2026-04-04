import { useCallback, useRef, useState } from 'react';
import type { ContainerSize } from './useContainerSize';

export interface TransformState {
  /** Current zoom level. 'fit' means auto-fit-to-viewport. */
  zoom: number;
  /** Pan offset in image-pixel coordinates. */
  panX: number;
  panY: number;
}

const MIN_ZOOM = 0.05;
const MAX_ZOOM = 32;
const ZOOM_FACTOR = 1.15; // per wheel tick

/**
 * Computes the fit-to-viewport zoom for a given image and container.
 */
export function fitZoom(
  containerW: number,
  containerH: number,
  imageW: number,
  imageH: number,
): number {
  if (containerW <= 0 || containerH <= 0 || imageW <= 0 || imageH <= 0) return 1;
  return Math.min(containerW / imageW, containerH / imageH);
}

/**
 * Returns a CSS transform string for the given state.
 * The canvas is positioned so that at fitZoom it's centered in the container.
 */
export function computeTransformCSS(
  state: TransformState,
  container: ContainerSize,
  imageW: number,
  imageH: number,
): React.CSSProperties {
  const { zoom, panX, panY } = state;
  const scaledW = imageW * zoom;
  const scaledH = imageH * zoom;
  // Center offset — when image is smaller than container, center it
  const offsetX = (container.width - scaledW) / 2 + panX * zoom;
  const offsetY = (container.height - scaledH) / 2 + panY * zoom;

  return {
    position: 'absolute' as const,
    left: 0,
    top: 0,
    width: imageW,
    height: imageH,
    transformOrigin: '0 0',
    transform: `translate(${offsetX}px, ${offsetY}px) scale(${zoom})`,
    imageRendering: zoom >= 4 ? 'pixelated' : undefined,
  };
}

/**
 * Formats zoom level for display.
 */
export function formatZoom(zoom: number, fitZoomVal: number): string {
  if (Math.abs(zoom - fitZoomVal) < 0.001) return 'Fit';
  return `${Math.round(zoom * 100)}%`;
}

/**
 * Hook managing pan/zoom state for the canvas viewport.
 */
export function useCanvasTransform(
  container: ContainerSize,
  imageW: number,
  imageH: number,
) {
  const fit = fitZoom(container.width, container.height, imageW, imageH);
  const [state, setState] = useState<TransformState>({ zoom: 0, panX: 0, panY: 0 });

  // Use fit zoom when state.zoom is 0 (initial) or when explicitly reset
  const effectiveZoom = state.zoom <= 0 ? fit : state.zoom;

  const isPanning = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  const resetToFit = useCallback(() => {
    setState({ zoom: 0, panX: 0, panY: 0 });
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      setState((prev) => {
        const oldZoom = prev.zoom <= 0 ? fit : prev.zoom;
        const direction = e.deltaY < 0 ? 1 : -1;
        const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, oldZoom * Math.pow(ZOOM_FACTOR, direction)));

        // Zoom centered on cursor position within the container
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const cursorX = e.clientX - rect.left;
        const cursorY = e.clientY - rect.top;

        // Current image-space point under cursor
        const scaledW = imageW * oldZoom;
        const scaledH = imageH * oldZoom;
        const offsetX = (container.width - scaledW) / 2 + prev.panX * oldZoom;
        const offsetY = (container.height - scaledH) / 2 + prev.panY * oldZoom;

        const imgX = (cursorX - offsetX) / oldZoom;
        const imgY = (cursorY - offsetY) / oldZoom;

        // New offset needed to keep the same image point under cursor
        const newScaledW = imageW * newZoom;
        const newScaledH = imageH * newZoom;
        const newBaseOffsetX = (container.width - newScaledW) / 2;
        const newBaseOffsetY = (container.height - newScaledH) / 2;

        const newPanX = (cursorX - newBaseOffsetX - imgX * newZoom) / newZoom;
        const newPanY = (cursorY - newBaseOffsetY - imgY * newZoom) / newZoom;

        return { zoom: newZoom, panX: newPanX, panY: newPanY };
      });
    },
    [fit, container.width, container.height, imageW, imageH],
  );

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return; // left click only
    isPanning.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    (e.currentTarget as HTMLElement).style.cursor = 'grabbing';
    e.preventDefault();
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isPanning.current) return;
      const dx = e.clientX - lastMouse.current.x;
      const dy = e.clientY - lastMouse.current.y;
      lastMouse.current = { x: e.clientX, y: e.clientY };

      setState((prev) => {
        const z = prev.zoom <= 0 ? fit : prev.zoom;
        return { ...prev, panX: prev.panX + dx / z, panY: prev.panY + dy / z };
      });
    },
    [fit],
  );

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    isPanning.current = false;
    (e.currentTarget as HTMLElement).style.cursor = '';
  }, []);

  const handleDoubleClick = useCallback(() => {
    resetToFit();
  }, [resetToFit]);

  return {
    state: { ...state, zoom: effectiveZoom },
    fitZoom: fit,
    resetToFit,
    handlers: {
      onWheel: handleWheel,
      onMouseDown: handleMouseDown,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp,
      onMouseLeave: handleMouseUp,
      onDoubleClick: handleDoubleClick,
    },
  };
}
