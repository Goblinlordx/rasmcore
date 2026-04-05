import { useCallback, useEffect, useRef, useState } from 'react';

export interface PreviewState {
  ready: boolean;
  processing: boolean;
  /** Last proxy render time in ms */
  proxyMs: number | null;
  /** Whether display mode is active (WebGPU direct rendering) */
  displayMode: boolean;
  /** Preview image dimensions (what the pixel buffer actually contains) */
  previewWidth: number;
  previewHeight: number;
}

export function usePreviewWorker() {
  const workerRef = useRef<Worker | null>(null);
  const readyRef = useRef(false);
  const processingRef = useRef(false);
  const [state, setState] = useState<PreviewState>({
    ready: false,
    processing: false,
    proxyMs: null,
    displayMode: false,
    previewWidth: 0,
    previewHeight: 0,
  });
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  /** External viewport canvas — proxy results draw here for main display */
  const viewportCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pendingLoadRef = useRef<ArrayBuffer | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const queuedChainRef = useRef<any[] | null>(null); // single-slot queue
  /** Called after each proxy render completes (for background warm trigger) */
  const onProxyCompleteRef = useRef<(() => void) | null>(null);
  /** OffscreenCanvas to transfer to worker for WebGPU display */
  const pendingDisplayCanvasRef = useRef<{ canvas: OffscreenCanvas; hdr: boolean } | null>(null);
  /** Ref-tracked display mode for use inside onmessage handler (avoids stale closure) */
  const displayModeRef = useRef(false);

  useEffect(() => {
    const w = new Worker(new URL('../v2-preview-worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.postMessage({ type: 'init' });

    w.onmessage = (e: MessageEvent) => {
      const { type } = e.data;

      if (type === 'ready') {
        readyRef.current = true;
        setState((s) => ({ ...s, ready: true }));
        // Send OffscreenCanvas for WebGPU display if queued
        if (pendingDisplayCanvasRef.current && workerRef.current) {
          const { canvas, hdr } = pendingDisplayCanvasRef.current;
          workerRef.current.postMessage(
            { type: 'init-display', canvas, hdr },
            [canvas],
          );
          displayModeRef.current = true;
          setState((s) => ({ ...s, displayMode: true }));
          pendingDisplayCanvasRef.current = null;
        }
        // Drain pending load if image was queued before SDK was ready
        if (pendingLoadRef.current && workerRef.current) {
          workerRef.current.postMessage({ type: 'load', imageBytes: pendingLoadRef.current }, [
            pendingLoadRef.current,
          ]);
          pendingLoadRef.current = null;
        }
        return;
      }

      if (type === 'loaded') {
        processingRef.current = false;
        setState((s) => ({
          ...s,
          processing: false,
          previewWidth: e.data.previewWidth ?? 0,
          previewHeight: e.data.previewHeight ?? 0,
        }));
        // Drain queue — a processChain may have been queued while loading
        if (queuedChainRef.current && workerRef.current) {
          const next = queuedChainRef.current;
          queuedChainRef.current = null;
          processingRef.current = true;
          setState((s) => ({ ...s, processing: true }));
          workerRef.current.postMessage({ type: 'process', chain: next });
        }
        return;
      }

      // WebGPU display mode — worker rendered directly to canvas, no PNG
      if (type === 'displayed') {
        processingRef.current = false;
        setState((s) => ({ ...s, processing: false, proxyMs: e.data.totalMs }));
        if (onProxyCompleteRef.current) onProxyCompleteRef.current();
        // Drain single-slot queue
        if (queuedChainRef.current && workerRef.current) {
          const next = queuedChainRef.current;
          queuedChainRef.current = null;
          processingRef.current = true;
          setState((s) => ({ ...s, processing: true }));
          workerRef.current.postMessage({ type: 'process', chain: next });
        }
        return;
      }

      if (type === 'result') {
        processingRef.current = false;
        setState((s) => ({ ...s, processing: false, proxyMs: e.data.totalMs }));

        const blob = new Blob([e.data.png], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          // Draw to sidebar preview canvas (editing thumbnail)
          const sidebar = previewCanvasRef.current;
          if (sidebar) {
            sidebar.width = img.width;
            sidebar.height = img.height;
            sidebar.getContext('2d')?.drawImage(img, 0, 0);
          }
          // Draw to main viewport canvas (proxy display).
          // Skip if in display mode — the canvas is owned by the worker's
          // WebGPU context, so getContext('2d') would return null.
          if (!displayModeRef.current) {
            const viewport = viewportCanvasRef.current;
            if (viewport && viewport.width > 0 && viewport.height > 0) {
              const ctx = viewport.getContext('2d');
              if (ctx) {
                ctx.drawImage(img, 0, 0, viewport.width, viewport.height);
              }
            }
          }
          URL.revokeObjectURL(url);

          // Notify that proxy render completed (triggers background warm)
          if (onProxyCompleteRef.current) onProxyCompleteRef.current();
        };
        img.src = url;

        // Drain single-slot queue
        if (queuedChainRef.current && workerRef.current) {
          const next = queuedChainRef.current;
          queuedChainRef.current = null;
          processingRef.current = true;
          setState((s) => ({ ...s, processing: true }));
          workerRef.current.postMessage({ type: 'process', chain: next });
        }
        return;
      }

      if (type === 'error') {
        processingRef.current = false;
        setState((s) => ({ ...s, processing: false }));
        console.warn('Preview worker error:', e.data.message);
        // Drain queue even on error
        if (queuedChainRef.current && workerRef.current) {
          const next = queuedChainRef.current;
          queuedChainRef.current = null;
          processingRef.current = true;
          setState((s) => ({ ...s, processing: true }));
          workerRef.current.postMessage({ type: 'process', chain: next });
        }
      }
    };

    return () => w.terminate();
  }, []);

  const loadImage = useCallback((imageBytes: ArrayBuffer) => {
    if (!workerRef.current) return;
    if (!readyRef.current) {
      pendingLoadRef.current = imageBytes;
      return;
    }
    processingRef.current = true;
    setState((s) => ({ ...s, processing: true }));
    workerRef.current.postMessage({ type: 'load', imageBytes }, [imageBytes]);
  }, []);

  // Single-slot queue: if busy, replace queued chain (latest params win)
  const processChain = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (chain: any[]) => {
      if (!workerRef.current || !readyRef.current) return;
      if (processingRef.current) {
        queuedChainRef.current = chain; // replace previous queued
        return;
      }
      processingRef.current = true;
      setState((s) => ({ ...s, processing: true }));
      workerRef.current.postMessage({ type: 'process', chain });
    },
    [],
  );

  /** Queue an OffscreenCanvas to be sent to the worker for WebGPU display. */
  const setDisplayCanvas = useCallback((canvas: OffscreenCanvas, hdr: boolean) => {
    if (readyRef.current && workerRef.current) {
      workerRef.current.postMessage(
        { type: 'init-display', canvas, hdr },
        [canvas],
      );
      setState((s) => ({ ...s, displayMode: true }));
    } else {
      pendingDisplayCanvasRef.current = { canvas, hdr };
    }
  }, []);

  /** Resize the OffscreenCanvas from the worker (for display mode). */
  const resizeCanvas = useCallback((width: number, height: number) => {
    workerRef.current?.postMessage({ type: 'resize-canvas', width, height });
  }, []);

  /** Forward viewport state to worker for shader-based pan/zoom. */
  const sendViewport = useCallback((
    panX: number, panY: number, zoom: number,
    canvasWidth: number, canvasHeight: number,
    imageWidth: number, imageHeight: number,
    toneMode?: number,
  ) => {
    workerRef.current?.postMessage({
      type: 'viewport',
      panX, panY, zoom,
      canvasWidth, canvasHeight,
      imageWidth, imageHeight,
      toneMode: toneMode ?? 0,
    });
  }, []);

  return {
    ...state,
    previewCanvasRef,
    viewportCanvasRef,
    onProxyCompleteRef,
    loadImage,
    processChain,
    setDisplayCanvas,
    resizeCanvas,
    sendViewport,
  };
}
