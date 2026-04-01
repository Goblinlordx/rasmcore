import { useCallback, useEffect, useRef, useState } from 'react';

export function usePreviewWorker() {
  const workerRef = useRef<Worker | null>(null);
  const readyRef = useRef(false);
  const processingRef = useRef(false);
  const [ready, setReady] = useState(false);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const w = new Worker(new URL('../preview-worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.postMessage({ type: 'init' });

    w.onmessage = (e: MessageEvent) => {
      const { type } = e.data;

      if (type === 'ready') {
        readyRef.current = true;
        setReady(true);
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
        return;
      }

      if (type === 'error') {
        processingRef.current = false;
        console.warn('Preview worker error:', e.data.message);
      }
    };

    return () => w.terminate();
  }, []);

  // Send load — fires and forgets (no queue needed for load)
  const loadImage = useCallback((imageBytes: ArrayBuffer) => {
    if (!workerRef.current || !readyRef.current) return;
    processingRef.current = true;
    workerRef.current.postMessage({ type: 'load', imageBytes }, [imageBytes]);
  }, []);

  // Send process — latest-wins (no queue, just overwrite)
  const processChain = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (chain: any[]) => {
      if (!workerRef.current || !readyRef.current) return;
      // Don't queue — just send. If worker is busy, it'll process the latest when done.
      workerRef.current.postMessage({ type: 'process', chain });
    },
    [],
  );

  return { ready, previewCanvasRef, loadImage, processChain };
}
