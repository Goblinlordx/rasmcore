import { useCallback, useEffect, useRef, useState } from 'react';

export interface WorkerTimings {
  totalMs: number;
  ops: { name: string; ms: number }[];
}

interface WorkerState {
  ready: boolean;
  processing: boolean;
  imageInfo: { width: number; height: number } | null;
  timings: WorkerTimings | null;
  error: string | null;
}

export function useWorker() {
  const workerRef = useRef<Worker | null>(null);
  const [state, setState] = useState<WorkerState>({
    ready: false,
    processing: false,
    imageInfo: null,
    timings: null,
    error: null,
  });
  const queueRef = useRef<unknown>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const originalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const thumbCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const onLoadedRef = useRef<(() => void) | null>(null);

  const clearProcessing = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setState((s) => ({ ...s, processing: false }));
  }, []);

  const drainQueue = useCallback(() => {
    if (queueRef.current && workerRef.current) {
      const next = queueRef.current as { imageBytes?: ArrayBuffer };
      queueRef.current = null;
      workerRef.current.postMessage(next, next.imageBytes ? [next.imageBytes] : []);
    }
  }, []);

  useEffect(() => {
    const w = new Worker(new URL('../pipeline-worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.postMessage({ type: 'init' });

    w.onmessage = (e: MessageEvent) => {
      const { type } = e.data;

      if (type === 'ready') {
        setState((s) => ({ ...s, ready: true }));
        return;
      }

      if (type === 'loaded') {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        setState((s) => ({
          ...s,
          processing: false,
          imageInfo: e.data.info,
          error: null,
        }));
        if (onLoadedRef.current) onLoadedRef.current();
        return;
      }

      if (type === 'result') {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        const blob = new Blob([e.data.png], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          const canvas =
            e.data.mode === 'thumb' ? thumbCanvasRef.current : previewCanvasRef.current;
          if (canvas) {
            canvas.width = img.width;
            canvas.height = img.height;
            canvas.getContext('2d')?.drawImage(img, 0, 0);
          }
          URL.revokeObjectURL(url);
        };
        img.src = url;

        if (e.data.mode === 'full' && e.data.timings) {
          setState((s) => ({
            ...s,
            processing: false,
            error: null,
            timings: { totalMs: e.data.totalMs, ops: e.data.timings },
          }));
        } else {
          setState((s) => ({ ...s, processing: false, error: null }));
        }
        // Drain queue
        if (queueRef.current) {
          const next = queueRef.current as { imageBytes?: ArrayBuffer };
          queueRef.current = null;
          setState((s) => ({ ...s, processing: true }));
          w.postMessage(next, next.imageBytes ? [next.imageBytes] : []);
        }
        return;
      }

      if (type === 'exported') {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        setState((s) => ({ ...s, processing: false }));
        const blob = new Blob([e.data.data], { type: e.data.mime });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `rasmcore-pipeline.${e.data.format}`;
        a.click();
        return;
      }

      if (type === 'error') {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        setState((s) => ({
          ...s,
          processing: false,
          error: e.data.message,
        }));
        // Drain queue even on error
        if (queueRef.current) {
          const next = queueRef.current as { imageBytes?: ArrayBuffer };
          queueRef.current = null;
          setState((s) => ({ ...s, processing: true }));
          w.postMessage(next, next.imageBytes ? [next.imageBytes] : []);
        }
      }
    };

    return () => {
      w.terminate();
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  const sendMessage = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (msg: any) => {
      if (!workerRef.current || !state.ready) return;
      if (state.processing) {
        queueRef.current = msg;
        return;
      }
      setState((s) => ({ ...s, processing: true, error: null }));
      // Safety timeout
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        setState((s) => {
          if (s.processing) {
            return { ...s, processing: false, error: 'Processing timed out' };
          }
          return s;
        });
      }, 15000);
      workerRef.current.postMessage(msg, msg.imageBytes ? [msg.imageBytes] : []);
    },
    [state.ready, state.processing],
  );

  return {
    ...state,
    sendMessage,
    clearProcessing,
    drainQueue,
    previewCanvasRef,
    originalCanvasRef,
    thumbCanvasRef,
    onLoadedRef,
  };
}
