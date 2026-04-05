import { useCallback, useEffect, useRef, useState } from 'react';
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

  // Native event target — set via gestureRef callback, triggers effects
  const [viewportEl, setViewportEl] = useState<HTMLElement | null>(null);

  // Touch state for pinch-to-zoom
  const lastTouchDist = useRef(0);
  const lastTouchMid = useRef({ x: 0, y: 0 });
  const touchCount = useRef(0);

  const resetToFit = useCallback(() => {
    setState({ zoom: 0, panX: 0, panY: 0 });
  }, []);

  // Wheel zoom — attached as native listener with { passive: false } to allow preventDefault.
  // React synthetic onWheel is passive and cannot preventDefault.
  const wheelHandlerRef = useRef(fit);
  wheelHandlerRef.current = fit;

  useEffect(() => {
    const el = viewportEl;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const currentFit = wheelHandlerRef.current;

      // Capture event values before setState (event may be recycled)
      const { deltaY, ctrlKey, clientX, clientY } = e;
      const rect = el.getBoundingClientRect();
      const cursorX = clientX - rect.left;
      const cursorY = clientY - rect.top;

      setState((prev) => {
        const oldZoom = prev.zoom <= 0 ? currentFit : prev.zoom;

        let newZoom: number;
        if (ctrlKey) {
          // Trackpad pinch (Chrome/Firefox): ctrlKey + fine-grained deltaY
          newZoom = oldZoom * Math.exp(-deltaY * 0.01);
        } else {
          // Mouse wheel: discrete ticks
          const direction = deltaY < 0 ? 1 : -1;
          newZoom = oldZoom * Math.pow(ZOOM_FACTOR, direction);
        }
        newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));

        // Zoom centered on cursor position within the container
        const scaledW = imageW * oldZoom;
        const scaledH = imageH * oldZoom;
        const offsetX = (container.width - scaledW) / 2 + prev.panX * oldZoom;
        const offsetY = (container.height - scaledH) / 2 + prev.panY * oldZoom;

        const imgX = (cursorX - offsetX) / oldZoom;
        const imgY = (cursorY - offsetY) / oldZoom;

        const newScaledW = imageW * newZoom;
        const newScaledH = imageH * newZoom;
        const newBaseOffsetX = (container.width - newScaledW) / 2;
        const newBaseOffsetY = (container.height - newScaledH) / 2;

        const newPanX = (cursorX - newBaseOffsetX - imgX * newZoom) / newZoom;
        const newPanY = (cursorY - newBaseOffsetY - imgY * newZoom) / newZoom;

        return { zoom: newZoom, panX: newPanX, panY: newPanY };
      });
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [viewportEl, container.width, container.height, imageW, imageH]);

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

  // ─── Touch handlers (pinch-to-zoom + single-finger pan) ──────────────

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const touches = e.touches;
    touchCount.current = touches.length;

    if (touches.length === 2) {
      // Pinch start — record initial distance and midpoint
      e.preventDefault();
      const dx = touches[1].clientX - touches[0].clientX;
      const dy = touches[1].clientY - touches[0].clientY;
      lastTouchDist.current = Math.hypot(dx, dy);
      lastTouchMid.current = {
        x: (touches[0].clientX + touches[1].clientX) / 2,
        y: (touches[0].clientY + touches[1].clientY) / 2,
      };
    } else if (touches.length === 1) {
      // Single finger pan start
      isPanning.current = true;
      lastMouse.current = { x: touches[0].clientX, y: touches[0].clientY };
    }
  }, []);

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      const touches = e.touches;

      if (touches.length === 2 && lastTouchDist.current > 0) {
        // Pinch zoom
        e.preventDefault();
        const dx = touches[1].clientX - touches[0].clientX;
        const dy = touches[1].clientY - touches[0].clientY;
        const dist = Math.hypot(dx, dy);
        const midX = (touches[0].clientX + touches[1].clientX) / 2;
        const midY = (touches[0].clientY + touches[1].clientY) / 2;
        const scale = dist / lastTouchDist.current;

        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const cursorX = midX - rect.left;
        const cursorY = midY - rect.top;

        // Also track pan from midpoint movement
        const panDx = midX - lastTouchMid.current.x;
        const panDy = midY - lastTouchMid.current.y;

        lastTouchDist.current = dist;
        lastTouchMid.current = { x: midX, y: midY };

        setState((prev) => {
          const oldZoom = prev.zoom <= 0 ? fit : prev.zoom;
          const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, oldZoom * scale));

          // Zoom centered on pinch midpoint
          const scaledW = imageW * oldZoom;
          const scaledH = imageH * oldZoom;
          const offsetX = (container.width - scaledW) / 2 + prev.panX * oldZoom;
          const offsetY = (container.height - scaledH) / 2 + prev.panY * oldZoom;

          const imgX = (cursorX - offsetX) / oldZoom;
          const imgY = (cursorY - offsetY) / oldZoom;

          const newScaledW = imageW * newZoom;
          const newScaledH = imageH * newZoom;
          const newBaseOffsetX = (container.width - newScaledW) / 2;
          const newBaseOffsetY = (container.height - newScaledH) / 2;

          const newPanX = (cursorX + panDx - newBaseOffsetX - imgX * newZoom) / newZoom;
          const newPanY = (cursorY + panDy - newBaseOffsetY - imgY * newZoom) / newZoom;

          return { zoom: newZoom, panX: newPanX, panY: newPanY };
        });
      } else if (touches.length === 1 && isPanning.current) {
        // Single finger pan
        const dx = touches[0].clientX - lastMouse.current.x;
        const dy = touches[0].clientY - lastMouse.current.y;
        lastMouse.current = { x: touches[0].clientX, y: touches[0].clientY };

        setState((prev) => {
          const z = prev.zoom <= 0 ? fit : prev.zoom;
          return { ...prev, panX: prev.panX + dx / z, panY: prev.panY + dy / z };
        });
      }
    },
    [fit, container.width, container.height, imageW, imageH],
  );

  const handleTouchEnd = useCallback((e: React.TouchEvent) => {
    const remaining = e.touches.length;
    if (remaining < 2) {
      lastTouchDist.current = 0;
    }
    if (remaining === 0) {
      isPanning.current = false;
    } else if (remaining === 1) {
      // Transition from pinch to single-finger pan
      isPanning.current = true;
      lastMouse.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
    touchCount.current = remaining;
  }, []);

  // ─── Safari GestureEvent support (trackpad pinch) ─────────────────────

  const gestureZoomBase = useRef(0);

  useEffect(() => {
    const el = viewportEl;
    if (!el || typeof (window as unknown as Record<string, unknown>).GestureEvent === 'undefined') return;

    const onGestureStart = (e: Event) => {
      e.preventDefault();
      gestureZoomBase.current = state.zoom <= 0 ? fit : state.zoom;
    };

    const onGestureChange = (e: Event) => {
      e.preventDefault();
      const ge = e as unknown as { scale: number; clientX: number; clientY: number };
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, gestureZoomBase.current * ge.scale));

      const rect = el.getBoundingClientRect();
      const cursorX = ge.clientX - rect.left;
      const cursorY = ge.clientY - rect.top;

      setState((prev) => {
        const oldZoom = prev.zoom <= 0 ? fit : prev.zoom;
        const scaledW = imageW * oldZoom;
        const scaledH = imageH * oldZoom;
        const offsetX = (container.width - scaledW) / 2 + prev.panX * oldZoom;
        const offsetY = (container.height - scaledH) / 2 + prev.panY * oldZoom;

        const imgX = (cursorX - offsetX) / oldZoom;
        const imgY = (cursorY - offsetY) / oldZoom;

        const newScaledW = imageW * newZoom;
        const newScaledH = imageH * newZoom;
        const newBaseOffsetX = (container.width - newScaledW) / 2;
        const newBaseOffsetY = (container.height - newScaledH) / 2;

        const newPanX = (cursorX - newBaseOffsetX - imgX * newZoom) / newZoom;
        const newPanY = (cursorY - newBaseOffsetY - imgY * newZoom) / newZoom;

        return { zoom: newZoom, panX: newPanX, panY: newPanY };
      });
    };

    const onGestureEnd = (e: Event) => { e.preventDefault(); };

    el.addEventListener('gesturestart', onGestureStart);
    el.addEventListener('gesturechange', onGestureChange);
    el.addEventListener('gestureend', onGestureEnd);
    return () => {
      el.removeEventListener('gesturestart', onGestureStart);
      el.removeEventListener('gesturechange', onGestureChange);
      el.removeEventListener('gestureend', onGestureEnd);
    };
  }, [viewportEl, fit, state.zoom, container.width, container.height, imageW, imageH]);

  /** Attach to the viewport element to enable native wheel + Safari gesture events. */
  const gestureRef = useCallback((el: HTMLElement | null) => {
    setViewportEl(el);
  }, []);

  const restore = useCallback((saved: TransformState) => {
    setState(saved);
  }, []);

  return {
    state: { ...state, zoom: effectiveZoom },
    fitZoom: fit,
    resetToFit,
    restore,
    gestureRef,
    handlers: {
      onMouseDown: handleMouseDown,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp,
      onMouseLeave: handleMouseUp,
      onDoubleClick: handleDoubleClick,
      onTouchStart: handleTouchStart,
      onTouchMove: handleTouchMove,
      onTouchEnd: handleTouchEnd,
    },
  };
}
