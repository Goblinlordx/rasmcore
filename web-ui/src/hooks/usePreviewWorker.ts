import { useCallback, useEffect, useRef, useState } from 'react';

export interface PreviewState {
  ready: boolean;
  processing: boolean;
  /** Last proxy render time in ms */
  proxyMs: number | null;
}

export function usePreviewWorker() {
  const workerRef = useRef<Worker | null>(null);
  const readyRef = useRef(false);
  const processingRef = useRef(false);
  const [state, setState] = useState<PreviewState>({
    ready: false,
    processing: false,
    proxyMs: null,
  });
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  /** External viewport canvas — proxy results draw here for main display */
  const viewportCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pendingLoadRef = useRef<ArrayBuffer | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const queuedChainRef = useRef<any[] | null>(null); // single-slot queue
  /** Called after each proxy render completes (for background warm trigger) */
  const onProxyCompleteRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const w = new Worker(new URL('../v2-preview-worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.postMessage({ type: 'init' });

    w.onmessage = (e: MessageEvent) => {
      const { type } = e.data;

      if (type === 'ready') {
        readyRef.current = true;
        setState((s) => ({ ...s, ready: true }));
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
        setState((s) => ({ ...s, processing: false }));
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
          // DON'T resize the canvas — keep it at the dimensions set by the
          // full-res worker or initial load. The pan/zoom transform depends on
          // the canvas matching imageWidth/imageHeight. Instead, stretch the
          // proxy image to fill the existing canvas size.
          const viewport = viewportCanvasRef.current;
          if (viewport && viewport.width > 0 && viewport.height > 0) {
            const ctx = viewport.getContext('2d');
            if (ctx) {
              ctx.drawImage(img, 0, 0, viewport.width, viewport.height);
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

  return {
    ...state,
    previewCanvasRef,
    viewportCanvasRef,
    onProxyCompleteRef,
    loadImage,
    processChain,
  };
}
