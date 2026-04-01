import { useCallback, useEffect, useRef, useState } from 'react';

export function usePreviewWorker() {
  const workerRef = useRef<Worker | null>(null);
  const readyRef = useRef(false);
  const processingRef = useRef(false);
  const [ready, setReady] = useState(false);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pendingLoadRef = useRef<ArrayBuffer | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const queuedChainRef = useRef<any[] | null>(null); // single-slot queue

  useEffect(() => {
    const w = new Worker(new URL('../preview-worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.postMessage({ type: 'init' });

    w.onmessage = (e: MessageEvent) => {
      const { type } = e.data;

      if (type === 'ready') {
        readyRef.current = true;
        setReady(true);
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
        return;
      }

      if (type === 'result') {
        processingRef.current = false;
        const blob = new Blob([e.data.png], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          const canvas = previewCanvasRef.current;
          if (canvas) {
            canvas.width = img.width;
            canvas.height = img.height;
            canvas.getContext('2d')?.drawImage(img, 0, 0);
          }
          URL.revokeObjectURL(url);
        };
        img.src = url;
        // Drain single-slot queue
        if (queuedChainRef.current && workerRef.current) {
          const next = queuedChainRef.current;
          queuedChainRef.current = null;
          processingRef.current = true;
          workerRef.current.postMessage({ type: 'process', chain: next });
        }
        return;
      }

      if (type === 'error') {
        processingRef.current = false;
        console.warn('Preview worker error:', e.data.message);
        // Drain queue even on error
        if (queuedChainRef.current && workerRef.current) {
          const next = queuedChainRef.current;
          queuedChainRef.current = null;
          processingRef.current = true;
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
      workerRef.current.postMessage({ type: 'process', chain });
    },
    [],
  );

  return { ready, previewCanvasRef, loadImage, processChain };
}
